use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ROOT: &str = "/dev/tenstorrent";
const DRAM_BANK_COUNT: usize = 8;
const WORKER_Y_RANGE: std::ops::Range<u8> = 2..12;

const P100_TENSIX_X: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14];
const P150_TENSIX_X: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
pub(crate) enum BoardKind {
    P100,
    P150,
}

impl BoardKind {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "p100" => Some(Self::P100),
            "p150" => Some(Self::P150),
            _ => None,
        }
    }

    fn infer(tensix_core_count: Option<usize>, active_dram_banks: Option<usize>) -> Option<Self> {
        match tensix_core_count {
            Some(0) => None,
            Some(count) if count <= 120 => Some(Self::P100),
            Some(count) if count <= 140 => Some(Self::P150),
            Some(_) => None,
            None => match active_dram_banks {
                Some(7) => Some(Self::P100),
                Some(8) => Some(Self::P150),
                _ => None,
            },
        }
    }

    fn config(self) -> BoardConfig {
        match self {
            Self::P100 => BoardConfig {
                name: "p100",
                tensix_x: &P100_TENSIX_X,
                prefetch: CoreCoord { x: 14, y: 2 },
                dispatch: CoreCoord { x: 14, y: 3 },
                default_tensix_core_count: 120,
                default_active_dram_banks: 7,
            },
            Self::P150 => BoardConfig {
                name: "p150",
                tensix_x: &P150_TENSIX_X,
                prefetch: CoreCoord { x: 16, y: 2 },
                dispatch: CoreCoord { x: 16, y: 3 },
                default_tensix_core_count: 140,
                default_active_dram_banks: 8,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoardConfig {
    pub(crate) name: &'static str,
    pub(crate) tensix_x: &'static [u8],
    pub(crate) prefetch: CoreCoord,
    pub(crate) dispatch: CoreCoord,
    pub(crate) default_tensix_core_count: usize,
    pub(crate) default_active_dram_banks: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeviceOverrides {
    pub(crate) board: Option<String>,
    pub(crate) tensix_core_count: Option<usize>,
    pub(crate) gddr_enabled_mask: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceInfo {
    pub(crate) id: usize,
    pub(crate) local_hardware_id: i32,
    pub(crate) path: PathBuf,
    pub(crate) board: Option<BoardKind>,
    pub(crate) arch: String,
    pub(crate) tensix_core_count: Option<usize>,
    pub(crate) worker_cores: Vec<CoreCoord>,
    pub(crate) prefetch_core: Option<CoreCoord>,
    pub(crate) dispatch_core: Option<CoreCoord>,
    pub(crate) harvested_dram_banks: Vec<usize>,
    pub(crate) active_dram_banks: usize,
}

impl DeviceInfo {
    pub(crate) fn discover() -> Vec<Self> {
        discover_with(Path::new(DEFAULT_ROOT), &|name| env::var(name).ok())
    }

    pub(crate) fn from_path(id: usize, path: PathBuf, overrides: DeviceOverrides) -> Self {
        let active_dram_banks_from_mask =
            overrides.gddr_enabled_mask.map(active_dram_banks_from_mask);
        let board = overrides
            .board
            .as_deref()
            .and_then(BoardKind::parse)
            .or_else(|| BoardKind::infer(overrides.tensix_core_count, active_dram_banks_from_mask));
        let board_config = board.map(BoardKind::config);
        let tensix_core_count = overrides
            .tensix_core_count
            .or_else(|| board_config.map(|config| config.default_tensix_core_count));
        let active_dram_banks = active_dram_banks_from_mask
            .or_else(|| board_config.map(|config| config.default_active_dram_banks))
            .unwrap_or(0);
        let harvested_dram_banks = overrides
            .gddr_enabled_mask
            .map(harvested_dram_banks_from_mask)
            .unwrap_or_default();
        let worker_cores = board_config
            .map(|config| worker_cores(config.tensix_x))
            .unwrap_or_default();
        let prefetch_core = board_config.map(|config| config.prefetch);
        let dispatch_core = board_config.map(|config| config.dispatch);
        let arch = board_config
            .map(|config| config.name.to_owned())
            .unwrap_or_else(|| "unknown".to_owned());

        Self {
            id,
            local_hardware_id: parse_local_hardware_id(&path).unwrap_or(id as i32),
            path,
            board,
            arch,
            tensix_core_count,
            worker_cores,
            prefetch_core,
            dispatch_core,
            harvested_dram_banks,
            active_dram_banks,
        }
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
        if !self.worker_cores.is_empty() {
            parts.push(format!("workers={}", self.worker_cores.len()));
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
        format!("Tenstorrent DRAM ({})", parts.join(", "))
    }

    pub(crate) fn memory_to_string(&self) -> String {
        format!("tt:{}:memory:{}", self.arch, self.id)
    }
}

fn discover_with<F>(root: &Path, env_lookup: &F) -> Vec<DeviceInfo>
where
    F: Fn(&str) -> Option<String>,
{
    let mut paths = Vec::new();

    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            paths.push(entry.path());
        }
    }

    paths.sort();
    paths
        .into_iter()
        .enumerate()
        .map(|(id, path)| DeviceInfo::from_path(id, path, overrides_for_device(id, env_lookup)))
        .collect()
}

fn overrides_for_device<F>(id: usize, env_lookup: &F) -> DeviceOverrides
where
    F: Fn(&str) -> Option<String>,
{
    DeviceOverrides {
        board: lookup_override(id, "TT_BOARD", env_lookup),
        tensix_core_count: lookup_override(id, "TT_TENSIX_CORES", env_lookup)
            .as_deref()
            .and_then(|value| value.trim().parse::<usize>().ok()),
        gddr_enabled_mask: lookup_override(id, "TT_GDDR_ENABLED", env_lookup)
            .as_deref()
            .and_then(parse_u32),
    }
}

fn lookup_override<F>(id: usize, key: &str, env_lookup: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    env_lookup(&format!("TT_DEVICE_{id}_{}", key.trim_start_matches("TT_")))
        .or_else(|| env_lookup(key))
}

fn parse_u32(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u32>().ok()
    }
}

fn parse_local_hardware_id(path: &Path) -> Option<i32> {
    path.file_name()?.to_str()?.parse().ok()
}

fn worker_cores(tensix_x: &[u8]) -> Vec<CoreCoord> {
    tensix_x
        .iter()
        .flat_map(|&x| WORKER_Y_RANGE.clone().map(move |y| CoreCoord { x, y }))
        .collect()
}

fn active_dram_banks_from_mask(mask: u32) -> usize {
    (mask & ((1u32 << DRAM_BANK_COUNT) - 1)).count_ones() as usize
}

fn harvested_dram_banks_from_mask(mask: u32) -> Vec<usize> {
    (0..DRAM_BANK_COUNT)
        .filter(|bank| ((mask >> bank) & 1) == 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_board_from_core_count() {
        assert_eq!(BoardKind::infer(Some(120), None), Some(BoardKind::P100));
        assert_eq!(BoardKind::infer(Some(130), None), Some(BoardKind::P150));
        assert_eq!(BoardKind::infer(Some(141), None), None);
    }

    #[test]
    fn infers_board_from_dram_bank_count_when_core_count_is_missing() {
        assert_eq!(BoardKind::infer(None, Some(7)), Some(BoardKind::P100));
        assert_eq!(BoardKind::infer(None, Some(8)), Some(BoardKind::P150));
    }

    #[test]
    fn builds_p100_device_metadata_from_overrides() {
        let device = DeviceInfo::from_path(
            2,
            PathBuf::from("/dev/tenstorrent/7"),
            DeviceOverrides {
                board: Some("p100".to_owned()),
                tensix_core_count: Some(120),
                gddr_enabled_mask: Some(0b0111_1111),
            },
        );

        assert_eq!(device.local_hardware_id, 7);
        assert_eq!(device.board, Some(BoardKind::P100));
        assert_eq!(device.arch, "p100");
        assert_eq!(device.tensix_core_count, Some(120));
        assert_eq!(device.worker_cores.len(), 120);
        assert_eq!(device.prefetch_core, Some(CoreCoord { x: 14, y: 2 }));
        assert_eq!(device.dispatch_core, Some(CoreCoord { x: 14, y: 3 }));
        assert_eq!(device.active_dram_banks, 7);
        assert_eq!(device.harvested_dram_banks, vec![7]);
    }

    #[test]
    fn per_device_overrides_win_over_global_overrides() {
        let env = |name: &str| match name {
            "TT_BOARD" => Some("p150".to_owned()),
            "TT_DEVICE_1_BOARD" => Some("p100".to_owned()),
            "TT_TENSIX_CORES" => Some("140".to_owned()),
            "TT_DEVICE_1_GDDR_ENABLED" => Some("0x7f".to_owned()),
            _ => None,
        };

        let devices = discover_with(Path::new("/tmp/does-not-exist"), &env);
        assert!(devices.is_empty());

        let overrides = overrides_for_device(1, &env);
        assert_eq!(overrides.board.as_deref(), Some("p100"));
        assert_eq!(overrides.tensix_core_count, Some(140));
        assert_eq!(overrides.gddr_enabled_mask, Some(0x7f));
    }
}
