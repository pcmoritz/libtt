use crate::executable::{Executable, Op, ValueDesc};
use crate::PJRT_Buffer_Type;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Default)]
struct Entry {
    count: u64,
    total_ns: u128,
    max_ns: u128,
}

#[derive(Default)]
struct State {
    execute_count: u64,
    execute_total_ns: u128,
    dispatch_finish_count: u64,
    dispatch_finish_total_ns: u128,
    program_launch_count: u64,
    program_sync_count: u64,
    executables: HashMap<String, Entry>,
    program_launches: HashMap<String, Entry>,
    program_syncs: HashMap<String, Entry>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

pub(crate) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub(crate) fn record_executable(plan: Option<&Executable>, start: Option<Instant>) {
    let Some(start) = start else {
        return;
    };
    if !enabled() {
        return;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let key = executable_key(plan);
    let mut state = state().lock().expect("profile mutex poisoned");
    state.execute_count += 1;
    state.execute_total_ns += elapsed_ns;
    add_entry(&mut state.executables, key, elapsed_ns);
    maybe_dump(&state, "execute");
}

pub(crate) fn record_dispatch_finish(start: Option<Instant>) {
    let Some(start) = start else {
        return;
    };
    if !enabled() {
        return;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let mut state = state().lock().expect("profile mutex poisoned");
    state.dispatch_finish_count += 1;
    state.dispatch_finish_total_ns += elapsed_ns;
}

pub(crate) fn record_program_launch(name: &str, staged: bool, start: Option<Instant>) {
    let Some(start) = start else {
        return;
    };
    if !enabled() {
        return;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let key = if staged {
        format!("staged:{name}")
    } else {
        format!("setup:{name}")
    };
    let mut state = state().lock().expect("profile mutex poisoned");
    state.program_launch_count += 1;
    add_entry(&mut state.program_launches, key, elapsed_ns);
}

pub(crate) fn record_program_sync(name: &str, staged: bool, start: Option<Instant>) {
    let Some(start) = start else {
        return;
    };
    if !enabled() {
        return;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let key = if staged {
        format!("staged:{name}")
    } else {
        format!("setup:{name}")
    };
    let mut state = state().lock().expect("profile mutex poisoned");
    state.program_sync_count += 1;
    add_entry(&mut state.program_syncs, key, elapsed_ns);
}

pub(crate) fn sync_each_program() -> bool {
    enabled() && matches!(env::var("LIBTT_PROFILE_SYNC_EACH_PROGRAM").as_deref(), Ok("1"))
}

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::default()))
}

fn enabled() -> bool {
    if let Some(trigger) = env::var_os("LIBTT_PROFILE_TRIGGER") {
        return !trigger.is_empty() && Path::new(&trigger).exists();
    }
    matches!(env::var("LIBTT_PROFILE").as_deref(), Ok("1"))
}

fn add_entry(entries: &mut HashMap<String, Entry>, key: String, elapsed_ns: u128) {
    let entry = entries.entry(key).or_default();
    entry.count += 1;
    entry.total_ns += elapsed_ns;
    entry.max_ns = entry.max_ns.max(elapsed_ns);
}

fn maybe_dump(state: &State, reason: &str) {
    let every = dump_every();
    if every == 0 || state.execute_count % every != 0 {
        return;
    }
    dump_state(state, reason);
}

fn dump_every() -> u64 {
    env::var("LIBTT_PROFILE_DUMP_EVERY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200)
}

fn top_n() -> usize {
    env::var("LIBTT_PROFILE_TOP_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20)
}

fn dump_state(state: &State, reason: &str) {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "[libtt-profile] reason={reason} execute_count={} execute_total_ms={:.3} dispatch_finish_count={} dispatch_finish_total_ms={:.3} program_launch_count={} program_sync_count={}",
        state.execute_count,
        ns_to_ms(state.execute_total_ns),
        state.dispatch_finish_count,
        ns_to_ms(state.dispatch_finish_total_ns),
        state.program_launch_count,
        state.program_sync_count,
    );
    append_top(&mut out, "executables", &state.executables);
    append_top(&mut out, "program_launch_host", &state.program_launches);
    append_top(&mut out, "program_sync", &state.program_syncs);
    eprint!("{out}");
    if let Some(path) = env::var_os("LIBTT_PROFILE_OUT").filter(|value| !value.is_empty()) {
        let _ = fs::write(path, out);
    }
}

fn append_top(out: &mut String, title: &str, entries: &HashMap<String, Entry>) {
    let mut rows = entries.iter().collect::<Vec<_>>();
    rows.sort_by(|(_, lhs), (_, rhs)| rhs.total_ns.cmp(&lhs.total_ns));
    let _ = writeln!(out, "[libtt-profile] top_{title}:");
    for (key, entry) in rows.into_iter().take(top_n()) {
        let _ = writeln!(
            out,
            "[libtt-profile] {title} total_ms={:.3} avg_ms={:.3} max_ms={:.3} count={} key={key}",
            ns_to_ms(entry.total_ns),
            ns_to_ms(entry.total_ns / u128::from(entry.count)),
            ns_to_ms(entry.max_ns),
            entry.count,
        );
    }
}

fn ns_to_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn executable_key(plan: Option<&Executable>) -> String {
    let Some(plan) = plan else {
        return "no_payload".to_owned();
    };
    let mut op_counts = BTreeMap::new();
    for op in &plan.ops {
        *op_counts.entry(op_family(op)).or_insert(0usize) += 1;
    }
    let ops = op_counts
        .iter()
        .map(|(op, count)| format!("{op}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let outputs = plan
        .output_ids
        .iter()
        .filter_map(|&id| plan.values.get(id as usize))
        .map(value_desc_key)
        .collect::<Vec<_>>()
        .join(",");
    format!("ops={} total_ops={} -> {outputs}", ops, plan.ops.len())
}

fn op_family(op: &Op) -> String {
    match op {
        Op::Parameter { .. } => "parameter".to_owned(),
        Op::Concatenate { dimension, .. } => format!("concatenate(dim={dimension})"),
        Op::Reshape { .. } => "reshape".to_owned(),
        Op::Slice { .. } => "slice".to_owned(),
        Op::Transpose { permutation, .. } => format!("transpose({permutation:?})"),
        Op::CustomCall {
            call_target_name, ..
        } => format!("custom_call({call_target_name})"),
        Op::Reduce {
            dimensions,
            reducer,
            ..
        } => format!("reduce({reducer:?},dims={dimensions:?})"),
        Op::ReduceWindow { reducer, .. } => format!("reduce_window({reducer:?})"),
        Op::Matmul {
            top_k_epilogue, ..
        } => {
            if top_k_epilogue.is_some() {
                "matmul_topk".to_owned()
            } else {
                "matmul".to_owned()
            }
        }
        Op::Constant { data, .. } => {
            if data.is_empty() {
                "constant_splat".to_owned()
            } else {
                format!("constant_data({}B)", data.len())
            }
        }
        Op::Select { .. } => "select".to_owned(),
        Op::BroadcastInDim {
            broadcast_dimensions,
            ..
        } => format!("broadcast({broadcast_dimensions:?})"),
        Op::Gather {
            slice_sizes,
            dimension_numbers,
            ..
        } => format!(
            "gather(axis={:?},slice={slice_sizes:?})",
            dimension_numbers.start_index_map
        ),
        Op::Scatter {
            dimension_numbers, ..
        } => format!(
            "scatter(dims={:?})",
            dimension_numbers.scatter_dims_to_operand_dims
        ),
        Op::Iota { iota_dimension, .. } => format!("iota(dim={iota_dimension})"),
        Op::TopK { k, .. } => format!("topk(k={k})"),
        Op::FusedElementwise {
            input_ids, nodes, ..
        } => {
            let op = nodes
                .last()
                .map(|node| format!("{:?}", node.kind))
                .unwrap_or_else(|| "empty".to_owned());
            format!("fused_{}_{}_{}", input_ids.len(), nodes.len(), op)
        }
    }
}

fn value_desc_key(desc: &ValueDesc) -> String {
    let dims = desc
        .dims
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join("x");
    format!("{}[{dims}]", dtype_key(desc.element_type))
}

fn dtype_key(dtype: PJRT_Buffer_Type) -> &'static str {
    match dtype {
        PJRT_Buffer_Type::PJRT_Buffer_Type_PRED => "pred",
        PJRT_Buffer_Type::PJRT_Buffer_Type_S8 => "s8",
        PJRT_Buffer_Type::PJRT_Buffer_Type_S16 => "s16",
        PJRT_Buffer_Type::PJRT_Buffer_Type_S32 => "s32",
        PJRT_Buffer_Type::PJRT_Buffer_Type_S64 => "s64",
        PJRT_Buffer_Type::PJRT_Buffer_Type_U8 => "u8",
        PJRT_Buffer_Type::PJRT_Buffer_Type_U16 => "u16",
        PJRT_Buffer_Type::PJRT_Buffer_Type_U32 => "u32",
        PJRT_Buffer_Type::PJRT_Buffer_Type_U64 => "u64",
        PJRT_Buffer_Type::PJRT_Buffer_Type_F16 => "f16",
        PJRT_Buffer_Type::PJRT_Buffer_Type_F32 => "f32",
        PJRT_Buffer_Type::PJRT_Buffer_Type_F64 => "f64",
        PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 => "bf16",
        _ => "other",
    }
}
