use std::fmt::Display;
use std::sync::OnceLock;

pub(crate) fn log(message: impl Display) {
    if enabled() {
        eprintln!("[libtt] {message}");
    }
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("LIBTT_LOG") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty()
                && normalized != "0"
                && normalized != "false"
                && normalized != "off"
        }
        Err(_) => false,
    })
}
