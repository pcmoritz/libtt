use crate::compiler::Compiler;
use crate::dispatch::{build_cq_launch, mcast_rects, DevMsgs, DispatchCommand};
use crate::hw::{align_down, align_up, noc_xy, Arc, CoreCoord, TensixL1, TensixMMIO};
use crate::kernels::kernel::RuntimeArgs;
use crate::linux::{NocOrdering, PinnedMemory, TlbWindow};
use crate::log::log;
use std::io;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

mod cq_bindings {
    #![allow(dead_code)]
    include!("cq_bindings.rs");
}

use cq_bindings::*;

const CQ_PREFETCH_Q_RD_PTR: usize = PREFETCH_Q_RD_PTR_ADDR as usize;
const CQ_PREFETCH_Q_PCIE_RD: usize = PREFETCH_Q_PCIE_RD_PTR_ADDR as usize;
const CQ_COMPLETION_WR_PTR: usize = DEV_COMPLETION_Q_WR_PTR as usize;
const CQ_COMPLETION_RD_PTR: usize = DEV_COMPLETION_Q_RD_PTR as usize;
const CQ_COMPLETION_Q0_EVENT: usize = DEV_COMPLETION_Q_WR_PTR as usize + 0x20;
const CQ_COMPLETION_Q1_EVENT: usize = DEV_COMPLETION_Q_WR_PTR as usize + 0x30;
const CQ_DISPATCH_SYNC_SEM: usize = DISPATCH_S_SYNC_SEM_BASE_ADDR as usize;
const CQ_PREFETCH_Q_BASE: usize = PREFETCH_Q_BASE as usize;
const CQ_PREFETCH_Q_ENTRY_SIZE: usize = size_of::<u16>();
const CQ_PREFETCH_Q_SIZE: usize = PREFETCH_Q_SIZE as usize;
const CQ_PREFETCH_Q_ENTRIES: usize = CQ_PREFETCH_Q_SIZE / CQ_PREFETCH_Q_ENTRY_SIZE;
const CQ_DISPATCH_CB_PAGES: u32 = DISPATCH_CB_PAGES;

const PCIE_NOC_BASE: u64 = 1 << 60;
const PCIE_ALIGN: usize = 64;
const L1_ALIGN: usize = 16;
const PAGE_SIZE: usize = 4096;

const HOST_ISSUE_BASE: usize = 4 * PCIE_ALIGN;
const HOST_ISSUE_SIZE: usize = 64 * 1024 * 1024;
const HOST_COMPLETION_BASE: usize = HOST_ISSUE_BASE + HOST_ISSUE_SIZE;
const HOST_COMPLETION_SIZE: usize = COMPLETION_QUEUE_SIZE as usize;
const HOST_TIMESTAMP_BASE: usize = HOST_COMPLETION_BASE + HOST_COMPLETION_SIZE;
const HOST_TIMESTAMP_STRIDE: usize = 16;
const HOST_TIMESTAMP_SLOTS: usize = 4096;
const HOST_TIMESTAMP_SIZE: usize = align_up(
    (HOST_TIMESTAMP_SLOTS * HOST_TIMESTAMP_STRIDE) as u64,
    PCIE_ALIGN as u64,
) as usize;
const HOST_PROFILER_BASE: usize = HOST_TIMESTAMP_BASE + HOST_TIMESTAMP_SIZE;
const HOST_CQ_WR_OFF: usize = HOST_COMPLETION_Q_WR_PTR as usize;
const HOST_CQ_RD_OFF: usize = 3 * PCIE_ALIGN;
const PCIE_BASE_GUARD_SIZE: usize = 1 << 30;

const CQ_DISPATCH_CMD_WRITE_LINEAR_HOST_IS_EVENT: u8 = 1;

const CQ_CMD_SIZE: usize = CQ_DISPATCH_CMD_SIZE as usize;
const DONE_STREAM: u16 = FIRST_STREAM_USED as u16;

pub(crate) struct FastDispatcher {
    path: PathBuf,
    prefetch_core: CoreCoord,
    dispatch_core: CoreCoord,
    _pcie_base_guard: PinnedMemory,
    cq_hw: CqSysmem,
    event_id: u32,
    runtime_template: Option<RuntimeCqTemplate>,
}

impl FastDispatcher {
    pub(crate) fn new(
        path: impl Into<PathBuf>,
        prefetch_core: CoreCoord,
        dispatch_core: CoreCoord,
        compiler: &Compiler,
    ) -> io::Result<Self> {
        // Match blackhole-py's fast-dispatch setup: reserve the base PCIe
        // sysmem window before allocating CQ sysmem, so CQ queues get a
        // nonzero local NOC offset.
        let path = path.into();
        let pcie_base_guard = new_sysmem(
            path.as_path(),
            PCIE_BASE_GUARD_SIZE,
            "reserve base PCIe sysmem window",
        )?;
        let mut cq_hw = CqSysmem::new(path.as_path(), prefetch_core, dispatch_core)?;
        start_dispatch_cores(&mut cq_hw, prefetch_core, dispatch_core, compiler)?;
        Ok(Self {
            path,
            prefetch_core,
            dispatch_core,
            _pcie_base_guard: pcie_base_guard,
            cq_hw,
            event_id: 0,
            runtime_template: None,
        })
    }

    pub(crate) fn execute(&mut self, commands: Vec<DispatchCommand>) -> io::Result<()> {
        let cq_commands = lower_ir(commands, go_word(self.dispatch_core));
        let mut queue = CommandQueue::default();
        queue.extend(cq_commands)?;
        self.event_id = self.event_id.wrapping_add(1);
        queue.append(CqCommand::HostEvent(self.event_id))?;
        self.cq_hw.flush(&queue)?;
        self.cq_hw
            .wait_completion(self.event_id, Duration::from_secs(10))
    }

    pub(crate) fn execute_runtime(&mut self, runtime_args: &RuntimeArgs) -> io::Result<()> {
        let blob_size = uniform_blob_size(runtime_args.blobs())?;

        let template_matches = self
            .runtime_template
            .as_ref()
            .is_some_and(|template| template.matches(runtime_args, blob_size));
        if !template_matches {
            self.runtime_template = Some(RuntimeCqTemplate::new(
                runtime_args,
                blob_size,
                go_word(self.dispatch_core),
            )?);
        }
        let template = self
            .runtime_template
            .as_mut()
            .expect("runtime template was just initialized");
        template.patch_runtime_blobs(runtime_args.blobs())?;

        self.event_id = self.event_id.wrapping_add(1);
        for record in &template.records_before_runtime {
            self.cq_hw.issue_write(record)?;
        }
        self.cq_hw.issue_write(&template.runtime_record)?;
        for record in &template.records_after_runtime {
            self.cq_hw.issue_write(record)?;
        }
        let event_record = host_event_record(self.event_id)?;
        self.cq_hw.issue_write(&event_record)?;
        self.cq_hw
            .wait_completion(self.event_id, Duration::from_secs(10))
    }
}

#[derive(Clone, Debug)]
struct RuntimeCqTemplate {
    cores: Vec<CoreCoord>,
    blob_size: usize,
    runtime_blob_start: usize,
    runtime_blob_stride: usize,
    records_before_runtime: Vec<Vec<u8>>,
    runtime_record: Vec<u8>,
    records_after_runtime: Vec<Vec<u8>>,
}

impl RuntimeCqTemplate {
    fn new(runtime_args: &RuntimeArgs, blob_size: usize, go_word: u32) -> io::Result<Self> {
        let cores = runtime_args.cores().to_vec();
        let records_before_runtime = relayed_command_records([
            CqCommand::WritePackedLarge {
                rects: mcast_rects(&cores),
                addr: TensixL1::GO_MSG as usize,
                data: vec![0, 0, 0, DevMsgs::RUN_MSG_RESET_READ_PTR_FROM_HOST],
            },
            CqCommand::WritePackedLarge {
                rects: mcast_rects(&cores),
                addr: TensixL1::GO_MSG_INDEX as usize,
                data: vec![0; size_of::<u32>()],
            },
        ])?;

        let (mut runtime_payload, body_start, stride) =
            write_packed_payload_header(&cores, TensixL1::KERNEL_CONFIG_BASE as usize, blob_size)?;
        runtime_payload.resize(body_start + stride * cores.len(), 0);
        let runtime_record = relay_inline(&runtime_payload);
        let runtime_blob_start = CQ_CMD_SIZE + body_start;

        let records_after_runtime = relayed_command_records([
            CqCommand::SetGoSignalNocData {
                cores: cores.clone(),
            },
            CqCommand::WaitStream {
                stream: DONE_STREAM,
                count: 0,
                clear: true,
            },
            CqCommand::SendGoSignal {
                go_word,
                stream: DONE_STREAM,
                count: 0,
                num_unicast: cores.len() as u8,
            },
            CqCommand::WaitStream {
                stream: DONE_STREAM,
                count: cores.len() as u32,
                clear: true,
            },
        ])?;

        Ok(Self {
            cores,
            blob_size,
            runtime_blob_start,
            runtime_blob_stride: stride,
            records_before_runtime,
            runtime_record,
            records_after_runtime,
        })
    }

    fn matches(&self, runtime_args: &RuntimeArgs, blob_size: usize) -> bool {
        self.blob_size == blob_size && self.cores == runtime_args.cores()
    }

    fn patch_runtime_blobs(&mut self, blobs: &[Vec<u8>]) -> io::Result<()> {
        if blobs.len() != self.cores.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "packed write core/data length mismatch: {} != {}",
                    self.cores.len(),
                    blobs.len()
                ),
            ));
        }
        for (index, blob) in blobs.iter().enumerate() {
            if blob.len() != self.blob_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "packed write blobs must have a uniform size",
                ));
            }
            let start = self.runtime_blob_start + self.runtime_blob_stride * index;
            self.runtime_record[start..start + self.blob_size].copy_from_slice(blob);
        }
        Ok(())
    }
}

impl Drop for FastDispatcher {
    fn drop(&mut self) {
        let _ = halt_cores(&self.path, &[self.prefetch_core, self.dispatch_core]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CqCommand {
    WritePackedLarge {
        rects: Vec<(CoreCoord, CoreCoord)>,
        addr: usize,
        data: Vec<u8>,
    },
    WritePacked {
        cores: Vec<CoreCoord>,
        addr: usize,
        data: Vec<Vec<u8>>,
    },
    SetGoSignalNocData {
        cores: Vec<CoreCoord>,
    },
    SendGoSignal {
        go_word: u32,
        stream: u16,
        count: u32,
        num_unicast: u8,
    },
    WaitStream {
        stream: u16,
        count: u32,
        clear: bool,
    },
    HostEvent(u32),
}

impl CqCommand {
    fn to_records(&self) -> io::Result<Vec<Vec<u8>>> {
        match self {
            Self::WritePackedLarge { rects, addr, data } => {
                let padded = pad_to(data, L1_ALIGN);
                let mut records = Vec::new();
                for batch in rects.chunks(CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS as usize)
                {
                    let mut record = cq_hdr_write_packed_large(batch.len())?;
                    for &(start, end) in batch {
                        let (xy, count) = noc_mcast_xy(start, end);
                        CQDispatchWritePackedLargeSubCmd {
                            noc_xy_addr: xy,
                            addr: to_u32(*addr, "CQ write address")?,
                            length_minus1: u16::try_from(data.len().checked_sub(1).ok_or_else(
                                || {
                                    io::Error::new(
                                        io::ErrorKind::InvalidInput,
                                        "CQ write data must not be empty",
                                    )
                                },
                            )?)
                            .map_err(|_| io::Error::other("CQ write payload too large"))?,
                            num_mcast_dests: count,
                            flags: CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK as u8,
                        }
                        .encode(&mut record);
                    }
                    let padded_len = align_up(record.len() as u64, L1_ALIGN as u64) as usize;
                    pad_vec_to(&mut record, padded_len);
                    for _ in batch {
                        record.extend_from_slice(&padded);
                    }
                    records.push(record);
                }
                Ok(records)
            }
            Self::WritePacked { cores, addr, data } => {
                if cores.len() != data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "packed write core/data length mismatch: {} != {}",
                            cores.len(),
                            data.len()
                        ),
                    ));
                }
                let Some(size) = data.first().map(Vec::len) else {
                    return Ok(vec![]);
                };
                if data.iter().any(|blob| blob.len() != size) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "packed write blobs must have a uniform size",
                    ));
                }
                let (mut record, body_start, stride) =
                    write_packed_payload_header(cores, *addr, size)?;
                for (index, blob) in data.iter().enumerate() {
                    record.extend_from_slice(blob);
                    pad_vec_to(&mut record, body_start + stride * (index + 1));
                }
                Ok(vec![record])
            }
            Self::SetGoSignalNocData { cores } => {
                let mut record = cq_record(
                    CQDispatchCmdId::CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA as u8,
                    CQDispatchSetGoSignalNocDataCmd {
                        pad1: 0,
                        pad2: 0,
                        num_words: cores.len() as u32,
                    },
                );
                for core in cores {
                    put_u32(&mut record, noc_xy(core.x, core.y));
                }
                Ok(vec![record])
            }
            Self::SendGoSignal {
                go_word,
                stream,
                count,
                num_unicast,
            } => Ok(vec![cq_record(
                CQDispatchCmdId::CQ_DISPATCH_CMD_SEND_GO_SIGNAL as u8,
                CQDispatchGoSignalMcastCmd {
                    go_signal: *go_word,
                    multicast_go_offset: CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
                    num_unicast_txns: *num_unicast,
                    noc_data_start_index: 0,
                    wait_count: *count,
                    wait_stream: u32::from(*stream),
                },
            )]),
            Self::WaitStream {
                stream,
                count,
                clear,
            } => Ok(vec![cq_record(
                CQDispatchCmdId::CQ_DISPATCH_CMD_WAIT as u8,
                CQDispatchWaitCmd {
                    flags: (CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM
                        | if *clear {
                            CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM
                        } else {
                            0
                        }) as u8,
                    stream: *stream,
                    addr: 0,
                    count: *count,
                },
            )]),
            Self::HostEvent(event_id) => {
                let mut payload = Vec::new();
                put_u32(&mut payload, *event_id);
                pad_vec_to(&mut payload, L1_ALIGN);
                let mut record = cq_record(
                    CQDispatchCmdId::CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST as u8,
                    CQDispatchWriteHostCmd {
                        is_event: CQ_DISPATCH_CMD_WRITE_LINEAR_HOST_IS_EVENT,
                        pad1: 0,
                        pad2: 0,
                        length: (CQ_CMD_SIZE + payload.len()) as u64,
                    },
                );
                record.extend_from_slice(&payload);
                Ok(vec![record])
            }
        }
    }
}

#[derive(Default)]
struct CommandQueue {
    stream: Vec<u8>,
    sizes_16b: Vec<usize>,
}

impl CommandQueue {
    fn append(&mut self, cmd: CqCommand) -> io::Result<()> {
        for record in relayed_records(cmd)? {
            self.sizes_16b.push(record.len() / CQ_CMD_SIZE);
            self.stream.extend_from_slice(&record);
        }
        Ok(())
    }

    fn extend(&mut self, cmds: Vec<CqCommand>) -> io::Result<()> {
        for cmd in cmds {
            self.append(cmd)?;
        }
        Ok(())
    }
}

struct CqSysmem {
    sysmem: PinnedMemory,
    noc_local: u64,
    prefetch_win: TlbWindow,
    dispatch_win: TlbWindow,
    issue_wr: usize,
    prefetch_q_wr_idx: usize,
    completion_base_16b: u32,
    completion_page_16b: u32,
    completion_end_16b: u32,
    completion_rd_16b: u32,
    completion_rd_toggle: u32,
}

impl CqSysmem {
    fn new(path: &Path, prefetch_core: CoreCoord, dispatch_core: CoreCoord) -> io::Result<Self> {
        let sysmem = new_sysmem(
            path,
            align_up(HOST_PROFILER_BASE as u64, PAGE_SIZE as u64) as usize,
            "initialize CQ sysmem",
        )?;
        if (sysmem.noc_addr() & PCIE_NOC_BASE) != PCIE_NOC_BASE {
            return Err(io::Error::other(format!(
                "bad CQ sysmem NOC address: 0x{:x}",
                sysmem.noc_addr()
            )));
        }
        let noc_local = sysmem.noc_addr() - PCIE_NOC_BASE;
        if noc_local == 0 {
            return Err(io::Error::other(
                "CQ sysmem unexpectedly allocated at base PCIe NOC offset",
            ));
        }
        if noc_local > u64::from(u32::MAX) {
            return Err(io::Error::other(format!(
                "CQ sysmem NOC offset too large: 0x{noc_local:x}"
            )));
        }
        let mut prefetch_win = TlbWindow::open(path, Arc::TLB_SIZE_2M, false)?;
        prefetch_win.target(prefetch_core, None, 0, NocOrdering::Strict)?;
        let mut dispatch_win = TlbWindow::open(path, Arc::TLB_SIZE_2M, false)?;
        dispatch_win.target(dispatch_core, None, 0, NocOrdering::Strict)?;

        let completion_base_16b =
            (((noc_local + HOST_COMPLETION_BASE as u64) >> 4) as u32) & 0x7fff_ffff;
        let completion_page_16b = (PAGE_SIZE >> 4) as u32;
        let completion_end_16b = completion_base_16b + (HOST_COMPLETION_SIZE >> 4) as u32;

        prefetch_win.write32(
            CQ_PREFETCH_Q_RD_PTR,
            (CQ_PREFETCH_Q_BASE + CQ_PREFETCH_Q_SIZE) as u32,
        )?;
        prefetch_win.write32(
            CQ_PREFETCH_Q_PCIE_RD,
            (noc_local + HOST_ISSUE_BASE as u64) as u32,
        )?;
        prefetch_win.write(CQ_PREFETCH_Q_BASE, &vec![0; CQ_PREFETCH_Q_SIZE])?;

        let mut cq = Self {
            sysmem,
            noc_local,
            prefetch_win,
            dispatch_win,
            issue_wr: 0,
            prefetch_q_wr_idx: 0,
            completion_base_16b,
            completion_page_16b,
            completion_end_16b,
            completion_rd_16b: completion_base_16b,
            completion_rd_toggle: 0,
        };
        cq.sysmem.write32(HOST_CQ_WR_OFF, completion_base_16b)?;
        cq.sysmem.write32(HOST_CQ_RD_OFF, completion_base_16b)?;
        Ok(cq)
    }

    fn flush(&mut self, queue: &CommandQueue) -> io::Result<()> {
        let mut offset = 0usize;
        for &size_16b in &queue.sizes_16b {
            let size = size_16b * CQ_CMD_SIZE;
            self.issue_write(&queue.stream[offset..offset + size])?;
            offset += size;
        }
        Ok(())
    }

    fn wait_completion(&mut self, event_id: u32, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let wr_raw = self.sysmem.read32(HOST_CQ_WR_OFF)?;
            let wr_16b = wr_raw & 0x7fff_ffff;
            let wr_toggle = wr_raw >> 31;
            if wr_16b != self.completion_rd_16b || wr_toggle != self.completion_rd_toggle {
                let off = ((self.completion_rd_16b as u64) << 4)
                    .checked_sub(self.noc_local)
                    .ok_or_else(|| io::Error::other("CQ completion pointer underflow"))?
                    as usize;
                let got = self.sysmem.read32(off + CQ_CMD_SIZE)?;
                self.completion_rd_16b += self.completion_page_16b;
                if self.completion_rd_16b >= self.completion_end_16b {
                    self.completion_rd_16b = self.completion_base_16b;
                    self.completion_rd_toggle ^= 1;
                }
                let raw =
                    (self.completion_rd_16b & 0x7fff_ffff) | (self.completion_rd_toggle << 31);
                self.dispatch_win.write32(CQ_COMPLETION_RD_PTR, raw)?;
                self.sysmem.write32(HOST_CQ_RD_OFF, raw)?;
                if got != event_id {
                    return Err(io::Error::other(format!(
                        "CQ completion event mismatch: got {got}, expected {event_id}"
                    )));
                }
                return Ok(());
            }
            if Instant::now() > deadline {
                let host_wr = self.sysmem.read32(HOST_CQ_WR_OFF)?;
                let host_rd = self.sysmem.read32(HOST_CQ_RD_OFF)?;
                let pref_pcie = self.prefetch_win.read32(CQ_PREFETCH_Q_PCIE_RD).unwrap_or(0);
                log(format!(
                    "CQ timeout event={event_id} cq_wr=0x{host_wr:08x} cq_rd=0x{host_rd:08x} pref_pcie=0x{pref_pcie:08x}"
                ));
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("timeout waiting for CQ completion event {event_id}"),
                ));
            }
            thread::sleep(Duration::from_micros(200));
        }
    }

    fn issue_write(&mut self, record: &[u8]) -> io::Result<()> {
        self.issue_wr = align_up(self.issue_wr as u64, PCIE_ALIGN as u64) as usize;
        if self.issue_wr + record.len() > HOST_ISSUE_SIZE {
            self.issue_wr = 0;
        }
        let base = HOST_ISSUE_BASE + self.issue_wr;
        self.sysmem.as_mut_slice()[base..base + record.len()].copy_from_slice(record);
        self.issue_wr += record.len();

        let idx = self.prefetch_q_wr_idx;
        self.wait_prefetch_slot_free(idx, Duration::from_secs(1))?;
        let off = CQ_PREFETCH_Q_BASE + idx * CQ_PREFETCH_Q_ENTRY_SIZE;
        let size_16b = u16::try_from(record.len() / CQ_CMD_SIZE)
            .map_err(|_| io::Error::other("CQ record too large for prefetch slot"))?;
        self.prefetch_win.write(off, &size_16b.to_le_bytes())?;
        self.prefetch_q_wr_idx = (idx + 1) % CQ_PREFETCH_Q_ENTRIES;
        Ok(())
    }

    fn wait_prefetch_slot_free(&mut self, idx: usize, timeout: Duration) -> io::Result<()> {
        let off = CQ_PREFETCH_Q_BASE + idx * CQ_PREFETCH_Q_ENTRY_SIZE;
        let deadline = Instant::now() + timeout;
        loop {
            let bytes = self.prefetch_win.read(off, 2)?;
            if u16::from_le_bytes([bytes[0], bytes[1]]) == 0 {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timeout waiting for CQ prefetch queue slot",
                ));
            }
            thread::sleep(Duration::from_micros(50));
        }
    }
}

fn new_sysmem(path: &Path, size: usize, label: &str) -> io::Result<PinnedMemory> {
    PinnedMemory::new(path, size).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!(
                "{label} failed for {} size=0x{size:x}: {err}",
                path.display()
            ),
        )
    })
}

fn lower_ir(commands: Vec<DispatchCommand>, go_word: u32) -> Vec<CqCommand> {
    let mut result = Vec::new();
    for command in commands {
        match command {
            DispatchCommand::Write { cores, addr, data } => {
                result.push(CqCommand::WritePackedLarge {
                    rects: mcast_rects(&cores),
                    addr,
                    data,
                });
            }
            DispatchCommand::WritePacked { cores, addr, data } => {
                result.push(CqCommand::WritePacked { cores, addr, data });
            }
            DispatchCommand::Launch { cores } => {
                let core_count = cores.len();
                result.push(CqCommand::SetGoSignalNocData { cores });
                result.push(CqCommand::WaitStream {
                    stream: DONE_STREAM,
                    count: 0,
                    clear: true,
                });
                result.push(CqCommand::SendGoSignal {
                    go_word,
                    stream: DONE_STREAM,
                    count: 0,
                    num_unicast: core_count as u8,
                });
                result.push(CqCommand::WaitStream {
                    stream: DONE_STREAM,
                    count: core_count as u32,
                    clear: true,
                });
            }
        }
    }
    result
}

fn start_dispatch_cores(
    cq_hw: &mut CqSysmem,
    prefetch_core: CoreCoord,
    dispatch_core: CoreCoord,
    compiler: &Compiler,
) -> io::Result<()> {
    let kernels = compiler.compile_cq_kernels()?;
    let kernel_off = L1_ALIGN as u32 + 2 * L1_ALIGN as u32;
    let mut pref_img = vec![0; L1_ALIGN];
    pref_img.extend_from_slice(&CQ_DISPATCH_CB_PAGES.to_le_bytes());
    pref_img.resize(2 * L1_ALIGN, 0);
    pref_img.resize(3 * L1_ALIGN, 0);
    let pref_launch = build_cq_launch(kernel_off, 0, L1_ALIGN)?;

    let disp_img = vec![0; 3 * L1_ALIGN];
    let dispatch_brisc = kernels
        .get("dispatch_brisc")
        .ok_or_else(|| io::Error::other("missing dispatch_brisc CQ kernel"))?;
    let ncrisc_off = align_up(
        (kernel_off as usize + dispatch_brisc.xip.len()) as u64,
        L1_ALIGN as u64,
    ) as u32;
    let disp_launch = build_cq_launch(kernel_off, ncrisc_off, L1_ALIGN)?;

    upload_cq_core(
        &mut cq_hw.prefetch_win,
        prefetch_core,
        &pref_img,
        &pref_launch,
        &[(kernel_off, &kernels["prefetch_brisc"].xip)],
    )?;
    cq_hw
        .dispatch_win
        .target(dispatch_core, None, 0, NocOrdering::Strict)?;
    cq_hw
        .dispatch_win
        .write32(CQ_COMPLETION_WR_PTR, cq_hw.completion_base_16b)?;
    cq_hw
        .dispatch_win
        .write32(CQ_COMPLETION_RD_PTR, cq_hw.completion_base_16b)?;
    cq_hw.dispatch_win.write32(CQ_COMPLETION_Q0_EVENT, 0)?;
    cq_hw.dispatch_win.write32(CQ_COMPLETION_Q1_EVENT, 0)?;
    cq_hw
        .dispatch_win
        .write(CQ_DISPATCH_SYNC_SEM, &vec![0; 8 * L1_ALIGN])?;
    upload_cq_core(
        &mut cq_hw.dispatch_win,
        dispatch_core,
        &disp_img,
        &disp_launch,
        &[
            (kernel_off, &kernels["dispatch_brisc"].xip),
            (ncrisc_off, &kernels["dispatch_s_ncrisc"].xip),
        ],
    )
}

fn upload_cq_core(
    win: &mut TlbWindow,
    core: CoreCoord,
    image: &[u8],
    launch: &[u8],
    kernels: &[(u32, &Vec<u8>)],
) -> io::Result<()> {
    win.target(core, None, 0, NocOrdering::Strict)?;
    win.write(TensixL1::KERNEL_CONFIG_BASE as usize, image)?;
    for &(offset, xip) in kernels {
        win.write(TensixL1::KERNEL_CONFIG_BASE as usize + offset as usize, xip)?;
    }
    win.write(TensixL1::LAUNCH as usize, launch)?;
    win.write(
        TensixL1::GO_MSG as usize,
        &go_word(CoreCoord { x: 0, y: 0 }).to_le_bytes(),
    )
}

fn halt_cores(path: &Path, cores: &[CoreCoord]) -> io::Result<()> {
    if cores.is_empty() {
        return Ok(());
    }
    let mmio_base = align_down(TensixMMIO::RISCV_DEBUG_REG_SOFT_RESET_0, Arc::TLB_SIZE_2M).0;
    let reset_off = (TensixMMIO::RISCV_DEBUG_REG_SOFT_RESET_0 - mmio_base) as usize;
    let mut win = TlbWindow::open(path, Arc::TLB_SIZE_2M, false)?;
    for &core in cores {
        win.target(core, None, mmio_base, NocOrdering::Strict)?;
        win.write32(reset_off, TensixMMIO::SOFT_RESET_ALL)?;
    }
    Ok(())
}

fn go_word(master: CoreCoord) -> u32 {
    u32::from(DevMsgs::RUN_MSG_GO) << 24 | u32::from(master.y) << 16 | u32::from(master.x) << 8
}

fn noc_mcast_xy(start: CoreCoord, end: CoreCoord) -> (u32, u8) {
    let xy = (u32::from(end.y) << 18)
        | (u32::from(end.x) << 12)
        | (u32::from(start.y) << 6)
        | u32::from(start.x);
    let count = (end.x - start.x + 1) * (end.y - start.y + 1);
    (xy, count)
}

trait CqEncode {
    fn encode(self, out: &mut Vec<u8>);
}

fn cq_record(cmd_id: u8, body: impl CqEncode) -> Vec<u8> {
    let mut record = Vec::with_capacity(CQ_CMD_SIZE);
    record.push(cmd_id);
    body.encode(&mut record);
    pad_vec_to(&mut record, CQ_CMD_SIZE);
    record
}

impl CqEncode for CQPrefetchRelayInlineCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.dispatcher_type);
        put_u16(out, self.pad);
        put_u32(out, self.length);
        put_u32(out, self.stride);
    }
}

impl CqEncode for CQDispatchWriteHostCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.is_event);
        put_u16(out, self.pad1);
        put_u32(out, self.pad2);
        put_u64(out, self.length);
    }
}

impl CqEncode for CQDispatchWritePackedLargeSubCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u32(out, self.noc_xy_addr);
        put_u32(out, self.addr);
        put_u16(out, self.length_minus1);
        put_u8(out, self.num_mcast_dests);
        put_u8(out, self.flags);
    }
}

impl CqEncode for CQDispatchWritePackedCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.flags);
        put_u16(out, self.count);
        put_u16(out, self.write_offset_index);
        put_u16(out, self.size);
        put_u32(out, self.addr);
    }
}

impl CqEncode for CQDispatchWritePackedLargeCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.type_);
        put_u16(out, self.count);
        put_u16(out, self.alignment);
        put_u16(out, self.write_offset_index);
    }
}

impl CqEncode for CQDispatchWaitCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.flags);
        put_u16(out, self.stream);
        put_u32(out, self.addr);
        put_u32(out, self.count);
    }
}

impl CqEncode for CQDispatchGoSignalMcastCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u32(out, self.go_signal);
        put_u8(out, self.multicast_go_offset);
        put_u8(out, self.num_unicast_txns);
        put_u8(out, self.noc_data_start_index);
        put_u32(out, self.wait_count);
        put_u32(out, self.wait_stream);
    }
}

impl CqEncode for CQDispatchSetGoSignalNocDataCmd {
    fn encode(self, out: &mut Vec<u8>) {
        put_u8(out, self.pad1);
        put_u16(out, self.pad2);
        put_u32(out, self.num_words);
    }
}

fn relay_inline(payload: &[u8]) -> Vec<u8> {
    let stride = align_up((CQ_CMD_SIZE + payload.len()) as u64, PCIE_ALIGN as u64) as usize;
    let mut record = cq_record(
        CQPrefetchCmdId::CQ_PREFETCH_CMD_RELAY_INLINE as u8,
        CQPrefetchRelayInlineCmd {
            dispatcher_type: 0,
            pad: 0,
            length: payload.len() as u32,
            stride: stride as u32,
        },
    );
    record.extend_from_slice(payload);
    record.resize(stride, 0);
    record
}

fn relayed_records(cmd: CqCommand) -> io::Result<Vec<Vec<u8>>> {
    cmd.to_records().map(|records| {
        records
            .into_iter()
            .map(|payload| relay_inline(&payload))
            .collect()
    })
}

fn relayed_command_records(
    commands: impl IntoIterator<Item = CqCommand>,
) -> io::Result<Vec<Vec<u8>>> {
    let mut records = Vec::new();
    for command in commands {
        records.extend(relayed_records(command)?);
    }
    Ok(records)
}

fn host_event_record(event_id: u32) -> io::Result<Vec<u8>> {
    let mut records = relayed_records(CqCommand::HostEvent(event_id))?;
    records
        .pop()
        .ok_or_else(|| io::Error::other("host event did not produce a CQ record"))
}

fn write_packed_payload_header(
    cores: &[CoreCoord],
    addr: usize,
    blob_size: usize,
) -> io::Result<(Vec<u8>, usize, usize)> {
    let mut record = cq_record(
        CQDispatchCmdId::CQ_DISPATCH_CMD_WRITE_PACKED as u8,
        CQDispatchWritePackedCmd {
            flags: CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE as u8,
            count: u16::try_from(cores.len())
                .map_err(|_| io::Error::other("too many CQ packed writes"))?,
            write_offset_index: 0,
            size: u16::try_from(blob_size)
                .map_err(|_| io::Error::other("CQ packed write payload too large"))?,
            addr: to_u32(addr, "CQ packed write address")?,
        },
    );
    for core in cores {
        put_u32(&mut record, noc_xy(core.x, core.y));
    }
    let noc_len = CQ_CMD_SIZE + cores.len() * size_of::<u32>();
    let body_start = align_up(noc_len as u64, L1_ALIGN as u64) as usize;
    pad_vec_to(&mut record, body_start);
    let stride = align_up(blob_size as u64, L1_ALIGN as u64) as usize;
    Ok((record, body_start, stride))
}

fn uniform_blob_size(blobs: &[Vec<u8>]) -> io::Result<usize> {
    let size = blobs
        .first()
        .map(Vec::len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "runtime args are empty"))?;
    if blobs.iter().any(|blob| blob.len() != size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "packed write blobs must have a uniform size",
        ));
    }
    Ok(size)
}

fn cq_hdr_write_packed_large(count: usize) -> io::Result<Vec<u8>> {
    Ok(cq_record(
        CQDispatchCmdId::CQ_DISPATCH_CMD_WRITE_PACKED_LARGE as u8,
        CQDispatchWritePackedLargeCmd {
            type_: CQDispatchCmdPackedWriteLargeType::CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_PROGRAM_BINARIES as u8,
            count: u16::try_from(count).map_err(|_| io::Error::other("too many CQ subcommands"))?,
            alignment: L1_ALIGN as u16,
            write_offset_index: 0,
        },
    ))
}

fn pad_to(data: &[u8], align: usize) -> Vec<u8> {
    let mut padded = data.to_vec();
    let len = align_up(padded.len() as u64, align as u64) as usize;
    padded.resize(len, 0);
    padded
}

fn pad_vec_to(data: &mut Vec<u8>, len: usize) {
    if data.len() < len {
        data.resize(len, 0);
    }
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn to_u32(value: usize, what: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| io::Error::other(format!("{what} does not fit in u32")))
}
