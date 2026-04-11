#[cfg(target_os = "linux")]
mod imp;
#[cfg(not(target_os = "linux"))]
mod stub;

#[cfg(target_os = "linux")]
pub(crate) use imp::{NocOrdering, Sysmem, TlbWindow};
#[cfg(not(target_os = "linux"))]
pub(crate) use stub::{NocOrdering, Sysmem, TlbWindow};
