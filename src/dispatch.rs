use crate::compiler::{CompiledKernel, Compiler, CoreSelection, Program};
use crate::hw::{Arc, CoreCoord, TensixL1, align_up};
use crate::linux::{NocOrdering, TlbWindow};
use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const RUN_MSG_GO: u8 = 0x80;
const RUN_MSG_RESET_READ_PTR_FROM_HOST: u8 = 0xE0;
const RUN_MSG_DONE: u8 = 0x00;
const DISPATCH_MODE_HOST: u8 = 1;
const FAST_CQ_NUM_CIRCULAR_BUFFERS: u8 = 32;
const L1_ALIGN: u32 = 16;
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DispatchCommand {
    Write {
        cores: Vec<CoreCoord>,
        addr: usize,
        data: Vec<u8>,
    },
    Launch {
        cores: Vec<CoreCoord>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Role {
    cores: Vec<CoreCoord>,
    reader: Option<CompiledKernel>,
    writer: Option<CompiledKernel>,
}

pub(crate) fn build_dispatch_plan(
    compiler: &Compiler,
    available_cores: &[CoreCoord],
    program: &Program,
) -> io::Result<Vec<DispatchCommand>> {
    let writer = if program.writer_kernel.is_empty() {
        None
    } else {
        Some(compiler.compile_dataflow(&program.writer_kernel, "brisc", None)?)
    };
    let reader = if program.reader_kernel.is_empty() {
        None
    } else {
        Some(compiler.compile_dataflow(&program.reader_kernel, "ncrisc", None)?)
    };
    let compute = if program.compute_kernel.is_empty() {
        None
    } else {
        Some(compiler.compile_compute(&program.compute_kernel, program)?)
    };

    let (all_cores, roles) = build_roles(compiler, available_cores, program, &reader, &writer)?;
    if all_cores.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no worker cores selected for dispatch",
        ));
    }

    let rta_sizes = (
        program.writer_args.len() * size_of::<u32>(),
        program.reader_args.len() * size_of::<u32>(),
        program.compute_args.len() * size_of::<u32>(),
    );
    let rta_total = rta_sizes.0 + rta_sizes.1 + rta_sizes.2;
    let sem_off = align_up(rta_total as u64, L1_ALIGN as u64) as usize;
    let rta_blob = pack_rta(
        &program.writer_args,
        &program.reader_args,
        &program.compute_args,
        program.semaphores,
        sem_off,
    );

    let mut commands = vec![
        DispatchCommand::Write {
            cores: all_cores.clone(),
            addr: TensixL1::GO_MSG as usize,
            data: vec![0, 0, 0, RUN_MSG_RESET_READ_PTR_FROM_HOST],
        },
        DispatchCommand::Write {
            cores: all_cores.clone(),
            addr: TensixL1::GO_MSG_INDEX as usize,
            data: vec![0; size_of::<u32>()],
        },
    ];

    if !rta_blob.is_empty() {
        commands.push(DispatchCommand::Write {
            cores: all_cores.clone(),
            addr: TensixL1::KERNEL_CONFIG_BASE as usize,
            data: rta_blob,
        });
    }

    for role in roles {
        let (shared_addr, shared_blob, launch_blob) = build_payload(
            program,
            role.reader.as_ref(),
            role.writer.as_ref(),
            compute.as_ref(),
            rta_sizes,
            sem_off,
            0,
        )?;
        commands.push(DispatchCommand::Write {
            cores: role.cores.clone(),
            addr: TensixL1::LAUNCH as usize,
            data: launch_blob,
        });
        if !shared_blob.is_empty() {
            commands.push(DispatchCommand::Write {
                cores: role.cores,
                addr: shared_addr as usize,
                data: shared_blob,
            });
        }
    }

    commands.push(DispatchCommand::Launch {
        cores: all_cores.clone(),
    });

    Ok(commands)
}

pub(crate) fn execute_slow_dispatch(path: &Path, commands: &[DispatchCommand]) -> io::Result<()> {
    let mut win = TlbWindow::open(path, Arc::TLB_SIZE_2M, false)?;

    for command in commands {
        match command {
            DispatchCommand::Write { cores, addr, data } => {
                for (start, end) in mcast_rects(cores) {
                    win.target(start, Some(end), 0, NocOrdering::Strict)?;
                    win.write(*addr, data)?;
                }
            }
            DispatchCommand::Launch { cores } => {
                let go_blob = [0u8, 0u8, 0u8, RUN_MSG_GO];
                for (start, end) in mcast_rects(cores) {
                    win.target(start, Some(end), 0, NocOrdering::Strict)?;
                    win.write(TensixL1::GO_MSG as usize, &go_blob)?;
                }

                for core in cores {
                    win.target(*core, None, 0, NocOrdering::Strict)?;
                    let deadline = Instant::now() + LAUNCH_TIMEOUT;
                    loop {
                        if win.read(TensixL1::GO_MSG as usize + 3, 1)?[0] == RUN_MSG_DONE {
                            break;
                        }
                        if Instant::now() > deadline {
                            return Err(io::Error::new(
                                io::ErrorKind::TimedOut,
                                format!("timeout waiting for core {core}"),
                            ));
                        }
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            }
        }
    }

    Ok(())
}

fn build_roles(
    compiler: &Compiler,
    available_cores: &[CoreCoord],
    program: &Program,
    reader: &Option<CompiledKernel>,
    writer: &Option<CompiledKernel>,
) -> io::Result<(Vec<CoreCoord>, Vec<Role>)> {
    if let Some((rows, cols)) = &program.grid {
        if rows.is_empty() || cols.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "program grid must contain at least one row and one column",
            ));
        }

        let available = available_cores.iter().copied().collect::<BTreeSet<_>>();
        let mut all_cores = Vec::with_capacity(rows.len() * cols.len());
        for &x in cols {
            for &y in rows {
                let core = CoreCoord { x, y };
                if !available.contains(&core) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("grid core {core} is not available on this device"),
                    ));
                }
                all_cores.push(core);
            }
        }
        all_cores.sort_unstable();

        let reader_recv = if program.reader_recv_kernel.is_empty() {
            reader.clone()
        } else {
            Some(compiler.compile_dataflow(&program.reader_recv_kernel, "ncrisc", None)?)
        };
        let writer_recv = if program.writer_recv_kernel.is_empty() {
            writer.clone()
        } else {
            Some(compiler.compile_dataflow(&program.writer_recv_kernel, "brisc", None)?)
        };

        let top_left = vec![CoreCoord {
            x: cols[0],
            y: rows[0],
        }];
        let top_row = cols[1..]
            .iter()
            .map(|&x| CoreCoord { x, y: rows[0] })
            .collect::<Vec<_>>();
        let left_col = rows[1..]
            .iter()
            .map(|&y| CoreCoord { x: cols[0], y })
            .collect::<Vec<_>>();
        let mut interior = Vec::new();
        for &x in &cols[1..] {
            for &y in &rows[1..] {
                interior.push(CoreCoord { x, y });
            }
        }

        let mut roles = Vec::new();
        for role in [
            Role {
                cores: top_left,
                reader: reader.clone(),
                writer: writer.clone(),
            },
            Role {
                cores: top_row,
                reader: reader_recv.clone(),
                writer: writer.clone(),
            },
            Role {
                cores: left_col,
                reader: reader.clone(),
                writer: writer_recv.clone(),
            },
            Role {
                cores: interior,
                reader: reader_recv,
                writer: writer_recv,
            },
        ] {
            if !role.cores.is_empty() {
                roles.push(role);
            }
        }

        Ok((all_cores, roles))
    } else {
        let all_cores = match program.cores {
            CoreSelection::Count(count) => available_cores.iter().copied().take(count).collect(),
            CoreSelection::All => available_cores.to_vec(),
        };
        Ok((
            all_cores.clone(),
            vec![Role {
                cores: all_cores,
                reader: reader.clone(),
                writer: writer.clone(),
            }],
        ))
    }
}

fn pack_rta(
    writer_args: &[u32],
    reader_args: &[u32],
    compute_args: &[u32],
    semaphores: usize,
    sem_off: usize,
) -> Vec<u8> {
    let mut data = Vec::with_capacity(
        (writer_args.len() + reader_args.len() + compute_args.len()) * size_of::<u32>(),
    );
    for arg in writer_args
        .iter()
        .chain(reader_args.iter())
        .chain(compute_args.iter())
    {
        data.extend_from_slice(&arg.to_le_bytes());
    }

    if semaphores > 0 {
        if sem_off > data.len() {
            data.resize(sem_off, 0);
        }
        data.resize(sem_off + semaphores * L1_ALIGN as usize, 0);
    }

    data
}

fn build_payload(
    program: &Program,
    reader: Option<&CompiledKernel>,
    writer: Option<&CompiledKernel>,
    compute: Option<&(CompiledKernel, CompiledKernel, CompiledKernel)>,
    rta_sizes: (usize, usize, usize),
    sem_off: usize,
    host_assigned_id: u32,
) -> io::Result<(u32, Vec<u8>, Vec<u8>)> {
    let rta_offsets = [0usize, rta_sizes.0, rta_sizes.0 + rta_sizes.1];
    let local_cb_off = align_up(
        sem_off
            .checked_add(program.semaphores * L1_ALIGN as usize)
            .ok_or_else(|| io::Error::other("local CB offset overflow"))? as u64,
        L1_ALIGN as u64,
    ) as usize;
    let (local_cb_mask, cb_blob) = build_cb_blob(program)?;
    let remote_cb_off = local_cb_off
        .checked_add(cb_blob.len())
        .ok_or_else(|| io::Error::other("remote CB offset overflow"))?;
    let mut kernel_off = align_up(remote_cb_off as u64, L1_ALIGN as u64) as usize;
    let mut kernel_text_offsets = [0u32; 5];
    let mut enables = 0u32;
    let mut kernels = Vec::<(usize, &CompiledKernel)>::new();
    if let Some(writer) = writer {
        kernels.push((0, writer));
    }
    if let Some(reader) = reader {
        kernels.push((1, reader));
    }
    if let Some((trisc0, trisc1, trisc2)) = compute {
        kernels.push((2, trisc0));
        kernels.push((3, trisc1));
        kernels.push((4, trisc2));
    }

    for &(index, kernel) in &kernels {
        kernel_text_offsets[index] = to_u32(kernel_off, "kernel_text_offset")?;
        kernel_off = align_up(
            kernel_off
                .checked_add(kernel.xip.len())
                .ok_or_else(|| io::Error::other("kernel payload overflow"))? as u64,
            L1_ALIGN as u64,
        ) as usize;
        enables |= 1u32 << index;
    }

    let mut shared = vec![
        0u8;
        kernel_off
            .checked_sub(local_cb_off)
            .ok_or_else(|| io::Error::other("kernel payload underflow"))?
    ];
    shared[..cb_blob.len()].copy_from_slice(&cb_blob);
    for &(index, kernel) in &kernels {
        let start = kernel_text_offsets[index] as usize - local_cb_off;
        let end = start + kernel.xip.len();
        shared[start..end].copy_from_slice(&kernel.xip);
    }

    let shared_addr = TensixL1::KERNEL_CONFIG_BASE
        .checked_add(to_u32(local_cb_off, "shared payload address")?)
        .ok_or_else(|| io::Error::other("shared payload address overflow"))?;
    let launch_blob = serialize_launch(
        sem_off,
        local_cb_off,
        remote_cb_off,
        rta_offsets,
        kernel_text_offsets,
        local_cb_mask,
        enables,
        host_assigned_id,
    )?;
    Ok((shared_addr, shared, launch_blob))
}

fn build_cb_blob(program: &Program) -> io::Result<(u32, Vec<u8>)> {
    if program.cbs.is_empty() {
        return Ok((0, Vec::new()));
    }

    let mut mask = 0u32;
    for cb in &program.cbs {
        if cb.index >= 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("circular buffer index {} is out of range", cb.index),
            ));
        }
        mask |= 1u32 << cb.index;
    }

    let entries = (u32::BITS - mask.leading_zeros()) as usize;
    let mut blob = vec![0u8; entries * 16];
    let mut next_addr = TensixL1::DATA_BUFFER_SPACE_BASE;
    let mut shared_addrs = HashMap::<usize, u32>::new();

    for cb in &program.cbs {
        let page_size = to_u32(cb.dtype.tile_size(), "circular buffer page size")?;
        let size = page_size
            .checked_mul(to_u32(cb.tiles, "circular buffer tile count")?)
            .ok_or_else(|| io::Error::other("circular buffer size overflow"))?;
        let shared_with = match cb.index {
            16 => shared_addrs.get(&24).copied(),
            24 => shared_addrs.get(&16).copied(),
            _ => None,
        };
        let addr = shared_with.unwrap_or(next_addr);
        if shared_with.is_none() {
            next_addr = next_addr
                .checked_add(size)
                .ok_or_else(|| io::Error::other("circular buffer address overflow"))?;
        }
        shared_addrs.insert(cb.index, addr);

        let off = cb.index * 16;
        blob[off..off + 4].copy_from_slice(&addr.to_le_bytes());
        blob[off + 4..off + 8].copy_from_slice(&size.to_le_bytes());
        blob[off + 8..off + 12]
            .copy_from_slice(&to_u32(cb.tiles, "circular buffer tiles")?.to_le_bytes());
        blob[off + 12..off + 16].copy_from_slice(&page_size.to_le_bytes());
    }

    Ok((mask, blob))
}

fn serialize_launch(
    sem_off: usize,
    local_cb_off: usize,
    remote_cb_off: usize,
    rta_offsets: [usize; 3],
    kernel_text_offsets: [u32; 5],
    local_cb_mask: u32,
    enables: u32,
    host_assigned_id: u32,
) -> io::Result<Vec<u8>> {
    let sem_off = to_u16(sem_off, "sem_offset")?;
    let local_cb_off = to_u16(local_cb_off, "local_cb_offset")?;
    let remote_cb_off = to_u16(remote_cb_off, "remote_cb_offset")?;
    let writer_rta_off = to_u16(rta_offsets[0], "writer rta offset")?;
    let reader_rta_off = to_u16(rta_offsets[1], "reader rta offset")?;
    let compute_rta_off = to_u16(rta_offsets[2], "compute rta offset")?;

    let mut out = Vec::with_capacity(96);
    for _ in 0..3 {
        out.extend_from_slice(&TensixL1::KERNEL_CONFIG_BASE.to_le_bytes());
    }
    for _ in 0..3 {
        out.extend_from_slice(&sem_off.to_le_bytes());
    }
    out.extend_from_slice(&local_cb_off.to_le_bytes());
    out.extend_from_slice(&remote_cb_off.to_le_bytes());
    out.extend_from_slice(&writer_rta_off.to_le_bytes());
    out.extend_from_slice(&local_cb_off.to_le_bytes());
    out.extend_from_slice(&reader_rta_off.to_le_bytes());
    out.extend_from_slice(&local_cb_off.to_le_bytes());
    for _ in 0..3 {
        out.extend_from_slice(&compute_rta_off.to_le_bytes());
        out.extend_from_slice(&local_cb_off.to_le_bytes());
    }
    out.push(DISPATCH_MODE_HOST);
    out.push(0);
    for offset in kernel_text_offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&local_cb_mask.to_le_bytes());
    out.push(1);
    out.push(0);
    out.push(FAST_CQ_NUM_CIRCULAR_BUFFERS);
    out.push(0);
    out.extend_from_slice(&host_assigned_id.to_le_bytes());
    out.extend_from_slice(&enables.to_le_bytes());
    for _ in 0..5 {
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    debug_assert_eq!(out.len(), 96);
    Ok(out)
}

fn to_u16(value: usize, label: &str) -> io::Result<u16> {
    u16::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} does not fit in u16: {value}"),
        )
    })
}

fn to_u32(value: usize, label: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} does not fit in u32: {value}"),
        )
    })
}

// Packs a set of cores into non-overlapping axis-aligned rectangles so multicast
// writes can target each rectangle with a single TLB configuration.
pub(crate) fn mcast_rects(cores: &[CoreCoord]) -> Vec<(CoreCoord, CoreCoord)> {
    if cores.is_empty() {
        return Vec::new();
    }

    let mut remaining = cores.iter().copied().collect::<BTreeSet<_>>();
    let mut rects = Vec::new();

    while let Some(&start) = remaining.iter().next() {
        let x0 = start.x;
        let y0 = start.y;
        let mut x1 = x0;
        while remaining.contains(&CoreCoord { x: x1 + 1, y: y0 }) {
            x1 += 1;
        }

        let mut y1 = y0;
        loop {
            let next_y = y1 + 1;
            let full_row = (x0..=x1).all(|x| remaining.contains(&CoreCoord { x, y: next_y }));
            if !full_row {
                break;
            }
            y1 = next_y;
        }

        for x in x0..=x1 {
            for y in y0..=y1 {
                remaining.remove(&CoreCoord { x, y });
            }
        }

        rects.push((CoreCoord { x: x0, y: y0 }, CoreCoord { x: x1, y: y1 }));
    }

    rects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CBConfig, MathFidelity};
    use crate::dram::DType;

    fn dummy_kernel(fill: u8, len: usize) -> CompiledKernel {
        CompiledKernel {
            xip: vec![fill; len],
            xip_text_bytes: len,
            disassembly: String::new(),
            elf_bytes: None,
        }
    }

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn build_cb_blob_packs_and_shares_buffers() {
        let program = Program {
            cbs: vec![
                CBConfig {
                    index: 0,
                    dtype: DType::Float16,
                    tiles: 2,
                },
                CBConfig {
                    index: 16,
                    dtype: DType::Float16B,
                    tiles: 1,
                },
                CBConfig {
                    index: 24,
                    dtype: DType::Float16B,
                    tiles: 1,
                },
            ],
            ..Program::default()
        };

        let (mask, blob) = build_cb_blob(&program).expect("cb blob");
        assert_eq!(mask, (1 << 0) | (1 << 16) | (1 << 24));
        assert_eq!(blob.len(), 25 * 16);
        assert_eq!(read_u32(&blob, 0), TensixL1::DATA_BUFFER_SPACE_BASE);
        assert_eq!(read_u32(&blob, 16 * 16), read_u32(&blob, 24 * 16));
    }

    #[test]
    fn build_payload_serializes_launch_message() {
        let writer = dummy_kernel(0x11, 31);
        let reader = dummy_kernel(0x22, 19);
        let compute = (
            dummy_kernel(0x33, 17),
            dummy_kernel(0x44, 18),
            dummy_kernel(0x55, 20),
        );
        let program = Program {
            cbs: vec![CBConfig::new(0, DType::Float16B)],
            writer_args: vec![1, 2],
            reader_args: vec![3],
            compute_args: vec![4, 5, 6],
            semaphores: 2,
            math_fidelity: MathFidelity::HiFi2,
            ..Program::default()
        };
        let rta_sizes = (8, 4, 12);
        let sem_off = align_up(
            (rta_sizes.0 + rta_sizes.1 + rta_sizes.2) as u64,
            L1_ALIGN as u64,
        ) as usize;

        let (shared_addr, shared, launch) = build_payload(
            &program,
            Some(&reader),
            Some(&writer),
            Some(&compute),
            rta_sizes,
            sem_off,
            7,
        )
        .expect("payload");

        assert_eq!(launch.len(), 96);
        assert_eq!(read_u32(&launch, 0), TensixL1::KERNEL_CONFIG_BASE);
        assert_eq!(read_u16(&launch, 12), sem_off as u16);
        assert_eq!(read_u16(&launch, 18), 64);
        assert_eq!(read_u16(&launch, 20), 80);
        assert_eq!(launch[42], DISPATCH_MODE_HOST);
        assert_eq!(read_u32(&launch, 44), 80);
        assert_eq!(read_u32(&launch, 48), 112);
        assert_eq!(read_u32(&launch, 52), 144);
        assert_eq!(read_u32(&launch, 56), 176);
        assert_eq!(read_u32(&launch, 60), 208);
        assert_eq!(read_u32(&launch, 64), 1);
        assert_eq!(launch[68], 1);
        assert_eq!(launch[70], FAST_CQ_NUM_CIRCULAR_BUFFERS);
        assert_eq!(read_u32(&launch, 72), 7);
        assert_eq!(read_u32(&launch, 76), 0b1_1111);
        assert_eq!(shared_addr, TensixL1::KERNEL_CONFIG_BASE + 64);
        assert_eq!(shared.len(), 176);
        assert_eq!(&shared[0..16], &build_cb_blob(&program).unwrap().1);
        assert_eq!(&shared[16..47], &writer.xip);
        assert_eq!(&shared[48..67], &reader.xip);
    }
}
