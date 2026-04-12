use crate::compiler::Compiler;
use crate::dram::{Allocator, DType, DramBuffer};
use crate::hw::{Arc, CoreCoord, Dram, DramTile, TensixMMIO, align_down, worker_cores};
use crate::linux::{NocOrdering, TlbWindow};
use crate::log::log;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_ROOT: &str = "/dev/tenstorrent";
const RUN_MSG_INIT: u8 = 0x40;
const RUN_MSG_DONE: u8 = 0x00;
const TENSIX_L1_SIZE: u32 = 0x180000;
const TENSIX_L1_GO_MSG: usize = 0x000370;
const TENSIX_L1_BRISC_FIRMWARE_BASE: u32 = 0x003840;
const TENSIX_L1_MEM_BANK_TO_NOC_SCRATCH: usize = 0x0116B0;
const BANK_PORT: [[u8; 2]; Dram::BANK_COUNT] = [
    [2, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
];

const P100_TENSIX_X: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14];
const P150_TENSIX_X: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoardConfig {
    pub(crate) name: &'static str,
    pub(crate) tensix_x: &'static [u8],
    pub(crate) prefetch: CoreCoord,
    pub(crate) dispatch: CoreCoord,
}

const P100: BoardConfig = BoardConfig {
    name: "p100",
    tensix_x: &P100_TENSIX_X,
    prefetch: CoreCoord { x: 14, y: 2 },
    dispatch: CoreCoord { x: 14, y: 3 },
};

const P150: BoardConfig = BoardConfig {
    name: "p150",
    tensix_x: &P150_TENSIX_X,
    prefetch: CoreCoord { x: 16, y: 2 },
    dispatch: CoreCoord { x: 16, y: 3 },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoardKind {
    P100,
    P150,
}

impl BoardKind {
    pub(crate) fn config(self) -> &'static BoardConfig {
        match self {
            Self::P100 => &P100,
            Self::P150 => &P150,
        }
    }

    fn from_tensix_core_count(core_count: usize) -> Option<Self> {
        match core_count {
            120 => Some(Self::P100),
            140 => Some(Self::P150),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProbeInfo {
    pub(crate) tensix_enabled_col_mask: u32,
    pub(crate) gddr_enabled_mask: u32,
}

pub struct Device {
    pub(crate) id: usize,
    pub(crate) local_hardware_id: usize,
    pub(crate) path: PathBuf,
    pub(crate) board: Option<BoardKind>,
    pub(crate) arch: String,
    pub(crate) tensix_core_count: Option<usize>,
    pub(crate) all_worker_cores: Vec<CoreCoord>,
    pub(crate) prefetch_core: Option<CoreCoord>,
    pub(crate) dispatch_core: Option<CoreCoord>,
    pub(crate) harvested_dram_banks: Vec<usize>,
    pub(crate) active_dram_banks: usize,
    pub(crate) dram_tiles: Vec<DramTile>,
    allocator: Option<Allocator>,
    compiler: Option<Compiler>,
}

impl Device {
    pub(crate) fn discover() -> Vec<Self> {
        discover_with(Path::new(DEFAULT_ROOT))
    }

    pub(crate) fn from_path(id: usize, path: PathBuf) -> Self {
        let local_hardware_id = local_hardware_id_from_path(&path).unwrap_or(id);
        Self::from_probe(
            id,
            local_hardware_id,
            path.clone(),
            detect_probe_info(&path),
        )
    }

    pub(crate) fn from_probe(
        id: usize,
        local_hardware_id: usize,
        path: PathBuf,
        probe: Option<ProbeInfo>,
    ) -> Self {
        let mut info = Self {
            id,
            local_hardware_id,
            path,
            board: None,
            arch: "unknown".to_owned(),
            tensix_core_count: None,
            all_worker_cores: Vec::new(),
            prefetch_core: None,
            dispatch_core: None,
            harvested_dram_banks: Vec::new(),
            active_dram_banks: 0,
            dram_tiles: Vec::new(),
            allocator: None,
            compiler: None,
        };

        if let Some(probe) = probe {
            let tensix_core_count = Arc::active_tensix_core_count(probe.tensix_enabled_col_mask);
            let board = BoardKind::from_tensix_core_count(tensix_core_count);
            let harvested_dram_banks = Dram::harvested_banks(probe.gddr_enabled_mask);
            let dram_tiles = Dram::tiles(&harvested_dram_banks);

            info.arch = board
                .map(|board| board.config().name.to_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            info.board = board;
            info.tensix_core_count = Some(tensix_core_count);
            info.active_dram_banks = Dram::active_banks(probe.gddr_enabled_mask);
            info.harvested_dram_banks = harvested_dram_banks;
            info.dram_tiles = dram_tiles;

            if let Some(board) = board {
                let config = board.config();
                info.all_worker_cores = worker_cores(config.tensix_x);
                info.prefetch_core = Some(config.prefetch);
                info.dispatch_core = Some(config.dispatch);
            }
        }

        info.initialize_compiler();
        info
    }

    pub(crate) fn cores(&self) -> Vec<CoreCoord> {
        self.all_worker_cores
            .iter()
            .copied()
            .filter(|core| Some(*core) != self.prefetch_core && Some(*core) != self.dispatch_core)
            .collect()
    }

    pub(crate) fn device_kind(&self) -> String {
        match self.board {
            Some(board) => format!("Tenstorrent {}", board.config().name),
            None => "Tenstorrent".to_owned(),
        }
    }

    pub(crate) fn device_debug_string(&self) -> String {
        let mut parts = vec![format!("board={}", self.arch)];
        if let Some(core_count) = self.tensix_core_count {
            parts.push(format!("cores={core_count}"));
        }
        if !self.all_worker_cores.is_empty() {
            parts.push(format!("workers={}", self.cores().len()));
        }
        if self.active_dram_banks > 0 {
            parts.push(format!("dram_banks={}", self.active_dram_banks));
        }
        if let (Some(prefetch), Some(dispatch)) = (self.prefetch_core, self.dispatch_core) {
            parts.push(format!("cq={prefetch}/{dispatch}"));
        }
        parts.push(format!("path={}", self.path.display()));
        format!("Tenstorrent device {} ({})", self.id, parts.join(", "))
    }

    pub(crate) fn device_to_string(&self) -> String {
        format!("tt:{}:{}", self.arch, self.id)
    }

    pub(crate) fn memory_debug_string(&self) -> String {
        let mut parts = vec![format!("device={}", self.id)];
        if self.active_dram_banks > 0 {
            parts.push(format!("dram_banks={}", self.active_dram_banks));
        }
        if !self.harvested_dram_banks.is_empty() {
            let harvested = self
                .harvested_dram_banks
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            parts.push(format!("harvested=[{harvested}]"));
        }
        if !self.dram_tiles.is_empty() {
            parts.push(format!("tiles={}", self.dram_tiles.len()));
        }
        format!("Tenstorrent DRAM ({})", parts.join(", "))
    }

    pub(crate) fn memory_to_string(&self) -> String {
        format!("tt:{}:memory:{}", self.arch, self.id)
    }

    pub fn local_hardware_id(&self) -> usize {
        self.local_hardware_id
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn open(local_hardware_id: usize) -> io::Result<Self> {
        Ok(load_device(local_hardware_id).1)
    }

    pub fn compiler(&self) -> Option<&Compiler> {
        self.compiler.as_ref()
    }

    pub fn alloc_write(
        &mut self,
        data: &[u8],
        dtype: DType,
        shape: &[usize],
        name: impl Into<String>,
    ) -> io::Result<DramBuffer> {
        let shape = shape.to_vec();
        let buffer = self
            .allocator_mut()?
            .alloc_for_host_data(data, dtype, shape, name)?;
        self.dram_write(&buffer, data)?;
        Ok(buffer)
    }

    pub fn dram_write(&mut self, buf: &DramBuffer, data: &[u8]) -> io::Result<()> {
        self.allocator_mut()?.write_host_data(buf, data)
    }

    pub fn dram_read(&mut self, buf: &DramBuffer) -> io::Result<Vec<u8>> {
        self.allocator_mut()?.read_host_data(buf)
    }

    fn allocator_mut(&mut self) -> io::Result<&mut Allocator> {
        if self.allocator.is_none() {
            self.allocator = Some(Allocator::from_device(self)?);
        }
        self.allocator
            .as_mut()
            .ok_or_else(|| io::Error::other("device allocator initialization failed"))
    }

    fn initialize_compiler(&mut self) {
        if self.compiler.is_some()
            || self.active_dram_banks == 0
            || self.all_worker_cores.is_empty()
            || self.prefetch_core.is_none()
            || self.dispatch_core.is_none()
        {
            return;
        }

        match Compiler::for_device(self) {
            Ok(compiler) => {
                self.compiler = Some(compiler);
                log(format!(
                    "device {} compiler initialized for {}",
                    self.id, self.arch
                ));
                if let Err(err) = self.upload_firmware() {
                    log(format!("device {} firmware upload skipped: {err}", self.id));
                }
            }
            Err(err) => {
                log(format!(
                    "device {} compiler initialization skipped: {err}",
                    self.id
                ));
            }
        }
    }

    pub fn upload_firmware(&mut self) -> io::Result<()> {
        let compiler = self
            .compiler
            .as_ref()
            .ok_or_else(|| io::Error::other("compiler has not been initialized"))?;
        let firmware = compiler.firmware();
        let all_cores = self.all_worker_cores.clone();
        if all_cores.is_empty() {
            return Err(io::Error::other("no worker cores discovered"));
        }

        let mmio_base =
            align_down(TensixMMIO::RISCV_DEBUG_REG_SOFT_RESET_0, Arc::TLB_SIZE_2M).0;
        let reset_off = (TensixMMIO::RISCV_DEBUG_REG_SOFT_RESET_0 - mmio_base) as usize;
        let mut staged = HashMap::<&str, Vec<(usize, Vec<u8>)>>::new();
        for name in ["brisc", "ncrisc", "trisc0", "trisc1", "trisc2"] {
            let compiled = firmware.get(name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing firmware image {name}"),
                )
            })?;
            let mut spans = Vec::new();
            for segment in &compiled.segments {
                if segment.data.is_empty() && segment.memsz == 0 {
                    continue;
                }
                let mut data = segment.data.clone();
                if segment.memsz as usize > data.len() {
                    data.resize(segment.memsz as usize, 0);
                }
                let mut addr = segment.paddr;
                if (TensixMMIO::LOCAL_RAM_START..=TensixMMIO::LOCAL_RAM_END).contains(&addr) {
                    addr = compiled.scratch_base + (addr - TensixMMIO::LOCAL_RAM_START);
                }
                if addr >= TENSIX_L1_SIZE {
                    return Err(io::Error::other(format!(
                        "{name}: bad paddr 0x{:x} -> 0x{addr:x}",
                        segment.paddr
                    )));
                }
                spans.push((addr as usize, data));
            }
            staged.insert(name, spans);
        }

        let jal = encode_jal_zero(TENSIX_L1_BRISC_FIRMWARE_BASE);
        let go_init = [0u8, 0u8, 0u8, RUN_MSG_INIT];
        let bank_table = build_bank_noc_table(&self.harvested_dram_banks, &all_cores)?;
        let rects = mcast_rects(&all_cores);

        let mut uc =
            TlbWindow::open(self.path.as_path(), all_cores[0], 0, Arc::TLB_SIZE_2M, false)?;
        let mut wc =
            TlbWindow::open(self.path.as_path(), all_cores[0], 0, Arc::TLB_SIZE_2M, true)?;

        for &(start, end) in &rects {
            uc.target(start, Some(end), mmio_base, NocOrdering::Strict)?;
            uc.write32(reset_off, TensixMMIO::SOFT_RESET_ALL)?;
        }

        for &(start, end) in &rects {
            wc.target(start, Some(end), 0, NocOrdering::Strict)?;
            for name in ["brisc", "ncrisc", "trisc0", "trisc1", "trisc2"] {
                for (addr, data) in staged.get(name).ok_or_else(|| {
                    io::Error::other(format!("missing staged firmware for {name}"))
                })? {
                    wc.write(*addr, data)?;
                }
            }
            wc.write(0, &jal)?;
            wc.write(TENSIX_L1_GO_MSG, &go_init)?;
            wc.write(TENSIX_L1_MEM_BANK_TO_NOC_SCRATCH, &bank_table)?;
        }

        let _ = wc.read32(0)?;

        let subordinate_reset_pcs = [
            (
                TensixMMIO::RISCV_DEBUG_REG_NCRISC_RESET_PC,
                firmware
                    .get("ncrisc")
                    .and_then(|fw| fw.text_base())
                    .ok_or_else(|| io::Error::other("ncrisc firmware missing text segment"))?,
            ),
            (
                TensixMMIO::RISCV_DEBUG_REG_TRISC0_RESET_PC,
                firmware
                    .get("trisc0")
                    .and_then(|fw| fw.text_base())
                    .ok_or_else(|| io::Error::other("trisc0 firmware missing text segment"))?,
            ),
            (
                TensixMMIO::RISCV_DEBUG_REG_TRISC1_RESET_PC,
                firmware
                    .get("trisc1")
                    .and_then(|fw| fw.text_base())
                    .ok_or_else(|| io::Error::other("trisc1 firmware missing text segment"))?,
            ),
            (
                TensixMMIO::RISCV_DEBUG_REG_TRISC2_RESET_PC,
                firmware
                    .get("trisc2")
                    .and_then(|fw| fw.text_base())
                    .ok_or_else(|| io::Error::other("trisc2 firmware missing text segment"))?,
            ),
        ];

        for &(start, end) in &rects {
            uc.target(start, Some(end), mmio_base, NocOrdering::Strict)?;
            for (reg, text_base) in subordinate_reset_pcs {
                uc.write32((reg - mmio_base) as usize, text_base)?;
            }
        }

        for &(start, end) in &rects {
            uc.target(start, Some(end), mmio_base, NocOrdering::Strict)?;
            uc.write32(reset_off, TensixMMIO::SOFT_RESET_BRISC_ONLY_RUN)?;
        }

        let probe = if all_cores.contains(&CoreCoord { x: 1, y: 2 }) {
            CoreCoord { x: 1, y: 2 }
        } else {
            all_cores[0]
        };
        uc.target(probe, None, 0, NocOrdering::Strict)?;
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if uc.read(TENSIX_L1_GO_MSG + 3, 1)?[0] == RUN_MSG_DONE {
                break;
            }
            if Instant::now() > deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("firmware not ready on {probe}"),
                ));
            }
            thread::sleep(Duration::from_millis(1));
        }

        log(format!("device {} firmware uploaded", self.id));
        Ok(())
    }
}

pub(crate) fn load_device(local_hardware_id: usize) -> (PathBuf, Device) {
    let path = PathBuf::from(format!("/dev/tenstorrent/{local_hardware_id}"));
    let info = Device::from_path(local_hardware_id, path.clone());
    (path, info)
}

fn detect_probe_info(path: &Path) -> Option<ProbeInfo> {
    match probe_info_for_device(path) {
        Ok(probe) => Some(probe),
        Err(err) => {
            log(format!("linux probe path={} failed: {err}", path.display()));
            None
        }
    }
}

fn probe_info_for_device(path: &Path) -> io::Result<ProbeInfo> {
    let (gddr_enabled_mask, tensix_enabled_col_mask) = read_arc_enabled_masks(path)?;
    log(format!(
        "linux probe path={} tensix_enabled_col_mask=0x{tensix_enabled_col_mask:08x} gddr_enabled_mask=0x{gddr_enabled_mask:08x}",
        path.display()
    ));

    Ok(ProbeInfo {
        tensix_enabled_col_mask,
        gddr_enabled_mask,
    })
}

fn read_arc_enabled_masks(path: &Path) -> io::Result<(u32, u32)> {
    let mut arc = TlbWindow::open(path, Arc::TILE, Arc::NOC_BASE, Arc::TLB_SIZE_2M, false)?;
    log(format!("linux probe opened {}", path.display()));
    let telemetry_ptr = arc.read32(Arc::SCRATCH_RAM_13)? as u64;
    let (csm_base, csm_offset) = align_down(telemetry_ptr, Arc::TLB_SIZE_2M);
    log(format!(
        "linux probe telemetry_ptr=0x{telemetry_ptr:x} csm_base=0x{csm_base:x} csm_offset=0x{csm_offset:x}"
    ));
    arc.target(Arc::TILE, None, csm_base, NocOrdering::Strict)?;

    let entry_count = arc.read32((csm_offset + 4) as usize)? as usize;
    log(format!("linux probe telemetry entry_count={entry_count}"));
    if entry_count == 0 || entry_count > 4096 {
        return Err(io::Error::other(format!(
            "invalid ARC telemetry entry_count {entry_count}"
        )));
    }

    let tags_base = csm_offset + 8;
    let data_base = tags_base + (entry_count as u64) * 4;
    let mut tensix_data_offset = None;
    let mut gddr_data_offset = None;

    for index in 0..entry_count {
        let tag_offset = arc.read32((tags_base + (index as u64) * 4) as usize)?;
        let tag = (tag_offset & 0xffff) as u16;
        let data_offset_words = (tag_offset >> 16) & 0xffff;

        if tag == Arc::TAG_TENSIX_ENABLED_COL {
            tensix_data_offset = Some(data_offset_words);
        } else if tag == Arc::TAG_GDDR_ENABLED {
            gddr_data_offset = Some(data_offset_words);
        }
    }

    let tensix_enabled_col_mask = match tensix_data_offset {
        Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
        None => Arc::DEFAULT_TENSIX_ENABLED,
    };
    let gddr_enabled_mask = match gddr_data_offset {
        Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
        None => Arc::DEFAULT_GDDR_ENABLED,
    };
    Ok((gddr_enabled_mask, tensix_enabled_col_mask))
}

fn encode_jal_zero(target: u32) -> [u8; 4] {
    (((target & 0xFF000) | ((target & 0x800) << 9) | ((target & 0x7FE) << 20) | 0x6F) as u32)
        .to_le_bytes()
}
fn discover_with(root: &Path) -> Vec<Device> {
    let mut paths = Vec::new();

    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            paths.push(entry.path());
        }
    }

    paths.sort();
    log(format!(
        "device discovery root={} entries={}",
        root.display(),
        paths.len()
    ));
    paths
        .into_iter()
        .enumerate()
        .map(|(id, path)| {
            log(format!("device[{id}] node={}", path.display()));
            Device::from_path(id, path)
        })
        .collect()
}

fn local_hardware_id_from_path(path: &Path) -> Option<usize> {
    path.file_name()?.to_str()?.parse().ok()
}

fn mcast_rects(cores: &[CoreCoord]) -> Vec<(CoreCoord, CoreCoord)> {
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

fn noc_xy(x: u8, y: u8) -> u16 {
    (((y as u16) << 6) | x as u16) & 0xFFFF
}

fn build_bank_noc_table(
    harvested_dram_banks: &[usize],
    worker_cores: &[CoreCoord],
) -> io::Result<Vec<u8>> {
    let num_dram_banks = Dram::BANK_COUNT - harvested_dram_banks.len();
    let num_l1_banks = worker_cores.len();

    let mut bank_xy = HashMap::<usize, (u8, u8)>::new();
    match harvested_dram_banks.len() {
        0 => {
            for bank in 0..Dram::BANK_COUNT {
                let x = if bank < 4 { 17 } else { 18 };
                bank_xy.insert(bank, (x, 12 + (bank % 4) as u8 * 3));
            }
        }
        1 => {
            let harvested = harvested_dram_banks[0];
            let half = 4usize;
            let mirror = if harvested < half {
                harvested + half - 1
            } else {
                harvested - half
            };

            let (left, right): (Vec<usize>, Vec<usize>) = if harvested < half {
                (
                    (half - 1..Dram::BANK_COUNT - 1)
                        .filter(|bank| *bank != mirror)
                        .chain(std::iter::once(mirror))
                        .collect(),
                    (0..half - 1).collect(),
                )
            } else {
                (
                    (0..half)
                        .filter(|bank| *bank != mirror)
                        .chain(std::iter::once(mirror))
                        .collect(),
                    (half..Dram::BANK_COUNT - 1).collect(),
                )
            };

            for (index, bank) in right.into_iter().enumerate() {
                bank_xy.insert(bank, (18, 12 + index as u8 * 3));
            }
            for (index, bank) in left.into_iter().enumerate() {
                bank_xy.insert(bank, (17, 12 + index as u8 * 3));
            }
        }
        count => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported harvested DRAM bank count: {count}"),
            ));
        }
    }

    let mut bytes = Vec::new();
    for noc in 0..2usize {
        for bank in 0..num_dram_banks {
            let (x, y0) = bank_xy.get(&bank).copied().ok_or_else(|| {
                io::Error::other(format!("missing NOC mapping for logical DRAM bank {bank}"))
            })?;
            bytes.extend_from_slice(&noc_xy(x, y0 + BANK_PORT[bank][noc]).to_le_bytes());
        }
    }

    let mut cols = worker_cores.iter().map(|core| core.x).collect::<Vec<_>>();
    cols.sort_unstable();
    cols.dedup();
    for _ in 0..2usize {
        for index in 0..num_l1_banks {
            let x = cols[index % cols.len()];
            let y = 2 + ((index / cols.len()) % 10) as u8;
            bytes.extend_from_slice(&noc_xy(x, y).to_le_bytes());
        }
    }

    for _ in 0..(num_dram_banks + num_l1_banks) {
        bytes.extend_from_slice(&0i32.to_le_bytes());
    }

    Ok(bytes)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn builds_minimal_device_metadata_from_path() {
        let device = Device::from_path(2, PathBuf::from("/dev/tenstorrent/7"));
        assert_eq!(device.local_hardware_id, 7);
        assert_eq!(device.arch, "unknown");
        assert_eq!(device.board, None);
        assert_eq!(device.tensix_core_count, None);
        assert!(device.all_worker_cores.is_empty());
        assert!(device.harvested_dram_banks.is_empty());
        assert_eq!(device.active_dram_banks, 0);
        assert!(device.dram_tiles.is_empty());
    }

    #[test]
    fn derives_blackhole_style_topology_from_probe_info() {
        let device = Device::from_probe(
            0,
            0,
            PathBuf::from("/dev/tenstorrent/0"),
            Some(ProbeInfo {
                tensix_enabled_col_mask: 0x0fff,
                gddr_enabled_mask: 0x7f,
            }),
        );

        assert_eq!(device.board, Some(BoardKind::P100));
        assert_eq!(device.arch, "p100");
        assert_eq!(device.tensix_core_count, Some(120));
        assert_eq!(device.all_worker_cores.len(), 120);
        assert_eq!(device.cores().len(), 118);
        assert_eq!(device.prefetch_core, Some(CoreCoord { x: 14, y: 2 }));
        assert_eq!(device.dispatch_core, Some(CoreCoord { x: 14, y: 3 }));
        assert_eq!(device.harvested_dram_banks, vec![7]);
        assert_eq!(device.active_dram_banks, 7);
        assert_eq!(device.dram_tiles.len(), 21);
        assert!(!device.cores().contains(&CoreCoord { x: 14, y: 2 }));
        assert!(!device.cores().contains(&CoreCoord { x: 14, y: 3 }));
    }

    #[test]
    fn derives_p150_worker_layout_from_probe_info() {
        let device = Device::from_probe(
            1,
            1,
            PathBuf::from("/dev/tenstorrent/1"),
            Some(ProbeInfo {
                tensix_enabled_col_mask: 0x3fff,
                gddr_enabled_mask: 0xff,
            }),
        );

        assert_eq!(device.board, Some(BoardKind::P150));
        assert_eq!(device.tensix_core_count, Some(140));
        assert_eq!(device.all_worker_cores.len(), 140);
        assert_eq!(device.cores().len(), 138);
        assert_eq!(device.active_dram_banks, 8);
        assert!(device.harvested_dram_banks.is_empty());
        assert_eq!(device.dram_tiles.len(), 24);
    }

    #[test]
    fn discovery_returns_empty_for_missing_root() {
        let devices = discover_with(Path::new("/tmp/does-not-exist"));
        assert!(devices.is_empty());
    }

    #[test]
    fn mcast_rects_groups_worker_cores_into_rectangles() {
        let rects = mcast_rects(&[
            CoreCoord { x: 1, y: 2 },
            CoreCoord { x: 2, y: 2 },
            CoreCoord { x: 1, y: 3 },
            CoreCoord { x: 2, y: 3 },
            CoreCoord { x: 4, y: 2 },
        ]);

        assert_eq!(
            rects,
            vec![
                (CoreCoord { x: 1, y: 2 }, CoreCoord { x: 2, y: 3 }),
                (CoreCoord { x: 4, y: 2 }, CoreCoord { x: 4, y: 2 }),
            ]
        );
    }

    #[test]
    fn bank_table_matches_expected_size_for_p100_layout() {
        let worker_cores = worker_cores(&P100_TENSIX_X);
        let table = build_bank_noc_table(&[7], &worker_cores).expect("bank table");
        let num_dram_banks = Dram::BANK_COUNT - 1;
        let num_l1_banks = worker_cores.len();
        let expected = 2 * num_dram_banks * size_of::<u16>()
            + 2 * num_l1_banks * size_of::<u16>()
            + (num_dram_banks + num_l1_banks) * size_of::<i32>();
        assert_eq!(table.len(), expected);
    }
}
