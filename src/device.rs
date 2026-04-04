use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_ROOT: &str = "/dev/tenstorrent";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceInfo {
    pub(crate) id: usize,
    pub(crate) local_hardware_id: i32,
    pub(crate) path: PathBuf,
    pub(crate) arch: String,
    pub(crate) tensix_core_count: Option<usize>,
    pub(crate) harvested_dram_banks: Vec<usize>,
    pub(crate) active_dram_banks: usize,
}

impl DeviceInfo {
    pub(crate) fn discover() -> Vec<Self> {
        discover_with(Path::new(DEFAULT_ROOT))
    }

    pub(crate) fn from_path(id: usize, path: PathBuf) -> Self {
        Self {
            id,
            local_hardware_id: parse_local_hardware_id(&path).unwrap_or(id as i32),
            path,
            arch: "unknown".to_owned(),
            tensix_core_count: None,
            harvested_dram_banks: Vec::new(),
            active_dram_banks: 0,
        }
    }

    pub(crate) fn device_kind(&self) -> String {
        "Tenstorrent".to_owned()
    }

    pub(crate) fn device_debug_string(&self) -> String {
        let mut parts = vec![format!("board={}", self.arch)];
        if let Some(core_count) = self.tensix_core_count {
            parts.push(format!("cores={core_count}"));
        }
        if self.active_dram_banks > 0 {
            parts.push(format!("dram_banks={}", self.active_dram_banks));
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

fn discover_with(root: &Path) -> Vec<DeviceInfo> {
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
        .map(|(id, path)| DeviceInfo::from_path(id, path))
        .collect()
}

fn parse_local_hardware_id(path: &Path) -> Option<i32> {
    path.file_name()?.to_str()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_minimal_device_metadata_from_path() {
        let device = DeviceInfo::from_path(2, PathBuf::from("/dev/tenstorrent/7"));
        assert_eq!(device.local_hardware_id, 7);
        assert_eq!(device.arch, "unknown");
        assert_eq!(device.tensix_core_count, None);
        assert!(device.harvested_dram_banks.is_empty());
        assert_eq!(device.active_dram_banks, 0);
    }

    #[test]
    fn discovery_returns_empty_for_missing_root() {
        let devices = discover_with(Path::new("/tmp/does-not-exist"));
        assert!(devices.is_empty());
    }
}
