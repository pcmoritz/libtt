use std::path::Path;

use super::ProbeInfo;

pub(super) fn detect_probe_info(path: &Path) -> Option<ProbeInfo> {
    let _ = path;
    None
}
