use std::path::Path;

use super::ProbeInfo;

pub(super) fn detect_probe_info(local_hardware_id: usize, path: &Path) -> Option<ProbeInfo> {
    let _ = (local_hardware_id, path);
    None
}
