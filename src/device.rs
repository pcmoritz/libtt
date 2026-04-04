use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
#[path = "device/linux.rs"]
mod probe_impl;
#[cfg(not(target_os = "linux"))]
#[path = "device/stub.rs"]
mod probe_impl;

const DEFAULT_ROOT: &str = "/dev/tenstorrent";
const ARC_DEFAULT_TENSIX_ENABLED: u32 = 0x3fff;
const DRAM_BANK_COUNT: usize = 8;
const WORKER_Y_START: u8 = 2;
const WORKER_Y_END: u8 = 12;

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
        if core_count == 0 {
            None
        } else if core_count <= 120 {
            Some(Self::P100)
        } else if core_count <= 140 {
            Some(Self::P150)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProbeInfo {
    pub(crate) arch: String,
    pub(crate) tensix_enabled_col_mask: u32,
    pub(crate) gddr_enabled_mask: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceInfo {
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
}

impl DeviceInfo {
    pub(crate) fn discover() -> Vec<Self> {
        discover_with(Path::new(DEFAULT_ROOT))
    }

    pub(crate) fn from_path(id: usize, path: PathBuf) -> Self {
        let local_hardware_id = local_hardware_id_from_path(&path).unwrap_or(id);
        Self::from_probe(
            id,
            local_hardware_id,
            path.clone(),
            detect_probe_info(local_hardware_id, &path),
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
        };

        if let Some(probe) = probe {
            let tensix_core_count = active_tensix_core_count(probe.tensix_enabled_col_mask);
            let board = select_core_layout(&probe.arch, tensix_core_count);
            let harvested_dram_banks = harvested_dram_banks(probe.gddr_enabled_mask);

            info.arch = if probe.arch.is_empty() {
                board.map(|board| board.config().name.to_owned())
                    .unwrap_or_else(|| "unknown".to_owned())
            } else {
                probe.arch
            };
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
}

fn detect_probe_info(local_hardware_id: usize, path: &Path) -> Option<ProbeInfo> {
    probe_impl::detect_probe_info(local_hardware_id, path)
}

fn discover_with(root: &Path) -> Vec<DeviceInfo> {
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
            DeviceInfo::from_path(id, path)
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

fn select_core_layout(arch: &str, tensix_core_count: usize) -> Option<BoardKind> {
    let normalized = arch.to_ascii_lowercase();
    if normalized.starts_with("p100") {
        Some(BoardKind::P100)
    } else if normalized.starts_with("p150") {
        if tensix_core_count == 120 {
            Some(BoardKind::P100)
        } else if tensix_core_count == 140 {
            Some(BoardKind::P150)
        } else {
            None
        }
    } else {
        BoardKind::from_tensix_core_count(tensix_core_count)
    }
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
    let harvested = harvested_dram_banks.iter().copied().collect::<std::collections::BTreeSet<_>>();
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

pub(super) fn log(message: impl AsRef<str>) {
    if log_enabled() {
        eprintln!("[libtt] {}", message.as_ref());
    }
}

fn log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("LIBTT_LOG") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && normalized != "0" && normalized != "false" && normalized != "off"
        }
        Err(_) => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_minimal_device_metadata_from_path() {
        let device = DeviceInfo::from_path(2, PathBuf::from("/dev/tenstorrent/7"));
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
        let device = DeviceInfo::from_probe(
            0,
            0,
            PathBuf::from("/dev/tenstorrent/0"),
            Some(ProbeInfo {
                arch: "p100".to_owned(),
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
        let device = DeviceInfo::from_probe(
            1,
            1,
            PathBuf::from("/dev/tenstorrent/1"),
            Some(ProbeInfo {
                arch: "p150".to_owned(),
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
    fn derives_p100_layout_for_harvested_p150() {
        let device = DeviceInfo::from_probe(
            0,
            0,
            PathBuf::from("/dev/tenstorrent/0"),
            Some(ProbeInfo {
                arch: "p150".to_owned(),
                tensix_enabled_col_mask: 0x0fff,
                gddr_enabled_mask: 0x7f,
            }),
        );

        assert_eq!(device.arch, "p150");
        assert_eq!(device.board, Some(BoardKind::P100));
        assert_eq!(device.prefetch_core, Some(CoreCoord { x: 14, y: 2 }));
        assert_eq!(device.dispatch_core, Some(CoreCoord { x: 14, y: 3 }));
    }

    #[test]
    fn discovery_returns_empty_for_missing_root() {
        let devices = discover_with(Path::new("/tmp/does-not-exist"));
        assert!(devices.is_empty());
    }
}
