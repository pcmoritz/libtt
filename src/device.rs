use crate::compiler::Compiler;
use crate::dram::{Allocator, DType, DramBuffer};
use crate::linux::{NocOrdering, TlbWindow};
use crate::log::log;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const DEFAULT_ROOT: &str = "/dev/tenstorrent";
const ARC_DEFAULT_TENSIX_ENABLED: u32 = 0x3fff;
const DEFAULT_GDDR_ENABLED: u32 = 0xff;
const DRAM_BANK_COUNT: usize = 8;
const WORKER_Y_START: u8 = 2;
const WORKER_Y_END: u8 = 12;
const ARC_TILE: CoreCoord = CoreCoord { x: 8, y: 0 };
const ARC_NOC_BASE: u64 = 0x8000_0000;
const ARC_SCRATCH_RAM_13: usize = 0x30434;
const TAG_TENSIX_ENABLED_COL: u16 = 34;
const TAG_GDDR_ENABLED: u16 = 36;
const TLB_SIZE_2M: u64 = 1 << 21;

const P100_TENSIX_X: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14];
const P150_TENSIX_X: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16];

const DRAM_BANK_TILE_YS: [[u8; 3]; DRAM_BANK_COUNT] = [
    [0, 1, 11],
    [2, 3, 10],
    [4, 8, 9],
    [5, 6, 7],
    [0, 1, 11],
    [2, 3, 10],
    [4, 8, 9],
    [5, 6, 7],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CoreCoord {
    pub(crate) x: u8,
    pub(crate) y: u8,
}

impl std::fmt::Display for CoreCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DramTile {
    pub(crate) bank: usize,
    pub(crate) x: u8,
    pub(crate) y: u8,
}

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
            let tensix_core_count = active_tensix_core_count(probe.tensix_enabled_col_mask);
            let board = BoardKind::from_tensix_core_count(tensix_core_count);
            let harvested_dram_banks = harvested_dram_banks(probe.gddr_enabled_mask);

            info.arch = board
                .map(|board| board.config().name.to_owned())
                .unwrap_or_else(|| "unknown".to_owned());
            info.board = board;
            info.tensix_core_count = Some(tensix_core_count);
            info.active_dram_banks = active_dram_banks(probe.gddr_enabled_mask);
            info.harvested_dram_banks = harvested_dram_banks.clone();
            info.dram_tiles = dram_tiles(&harvested_dram_banks);

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
            }
            Err(err) => {
                log(format!(
                    "device {} compiler initialization skipped: {err}",
                    self.id
                ));
            }
        }
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
    let mut arc = TlbWindow::open(path, ARC_TILE, ARC_NOC_BASE, TLB_SIZE_2M, false)?;
    log(format!("linux probe opened {}", path.display()));
    let telemetry_ptr = arc.read32(ARC_SCRATCH_RAM_13)? as u64;
    let (csm_base, csm_offset) = align_down(telemetry_ptr, TLB_SIZE_2M);
    log(format!(
        "linux probe telemetry_ptr=0x{telemetry_ptr:x} csm_base=0x{csm_base:x} csm_offset=0x{csm_offset:x}"
    ));
    arc.target(ARC_TILE, None, csm_base, NocOrdering::Strict)?;

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

        if tag == TAG_TENSIX_ENABLED_COL {
            tensix_data_offset = Some(data_offset_words);
        } else if tag == TAG_GDDR_ENABLED {
            gddr_data_offset = Some(data_offset_words);
        }
    }

    let tensix_enabled_col_mask = match tensix_data_offset {
        Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
        None => ARC_DEFAULT_TENSIX_ENABLED,
    };
    let gddr_enabled_mask = match gddr_data_offset {
        Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
        None => DEFAULT_GDDR_ENABLED,
    };
    Ok((gddr_enabled_mask, tensix_enabled_col_mask))
}

fn align_down(value: u64, alignment: u64) -> (u64, u64) {
    let base = value & !(alignment - 1);
    (base, value - base)
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

fn worker_cores(tensix_x: &[u8]) -> Vec<CoreCoord> {
    let mut cores = Vec::with_capacity(tensix_x.len() * (WORKER_Y_END - WORKER_Y_START) as usize);

    for &x in tensix_x {
        for y in WORKER_Y_START..WORKER_Y_END {
            cores.push(CoreCoord { x, y });
        }
    }

    cores
}

fn active_tensix_core_count(enabled_col_mask: u32) -> usize {
    (enabled_col_mask & ARC_DEFAULT_TENSIX_ENABLED).count_ones() as usize * 10
}

fn local_hardware_id_from_path(path: &Path) -> Option<usize> {
    path.file_name()?.to_str()?.parse().ok()
}

fn active_dram_banks(gddr_enabled_mask: u32) -> usize {
    (0..DRAM_BANK_COUNT)
        .filter(|bank| ((gddr_enabled_mask >> bank) & 1) != 0)
        .count()
}

fn harvested_dram_banks(gddr_enabled_mask: u32) -> Vec<usize> {
    (0..DRAM_BANK_COUNT)
        .filter(|bank| ((gddr_enabled_mask >> bank) & 1) == 0)
        .collect()
}

fn dram_tiles(harvested_dram_banks: &[usize]) -> Vec<DramTile> {
    let harvested = harvested_dram_banks
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut tiles = Vec::new();

    for bank in 0..DRAM_BANK_COUNT {
        if harvested.contains(&bank) {
            continue;
        }

        let x = dram_bank_x(bank);
        for &y in &DRAM_BANK_TILE_YS[bank] {
            tiles.push(DramTile { bank, x, y });
        }
    }

    tiles
}

fn dram_bank_x(bank: usize) -> u8 {
    if bank < 4 { 0 } else { 9 }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
