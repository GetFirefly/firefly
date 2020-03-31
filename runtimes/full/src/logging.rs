#[cfg(not(target_arch = "wasm32"))]
use colored::*;
#[cfg(not(any(test, target_arch = "wasm32")))]
use log::SetLoggerError;
use log::{Level, Log, Metadata, Record};

use crate::time::system;
use crate::time::Unit::Second;

pub struct Logger {
    level: Level,
    color: bool,
}

impl Logger {
    #[cfg(not(any(test, target_arch = "wasm32")))]
    pub fn init(level: Level) -> Result<(), SetLoggerError> {
        log::set_max_level(level.to_level_filter());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn log_color(record: &Record) {
        let level = match record.level() {
            Level::Error => record.level().to_string().red(),
            Level::Warn => record.level().to_string().yellow(),
            Level::Info => record.level().to_string().cyan(),
            Level::Debug => record.level().to_string().purple(),
            Level::Trace => record.level().to_string().normal(),
        };
        println!(
            "{} {:<5} [{}] {}",
            system::time(Second),
            level,
            record.module_path().unwrap_or_default(),
            record.args(),
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn log_plain(record: &Record) {
        println!(
            "{} {:<5} [{}] {}",
            system::time(Second),
            record.level(),
            record.module_path().unwrap_or_default(),
            record.args(),
        )
    }

    #[cfg(target_arch = "wasm32")]
    fn log_color(record: &Record) {
        Self::log_plain(record);
    }

    #[cfg(target_arch = "wasm32")]
    fn log_plain(record: &Record) {
        let msg = format!(
            "{} {:<5} [{}] {}",
            system::time(Second),
            record.level(),
            record.module_path().unwrap_or_default(),
            record.args()
        );
        crate::sys::io::puts(&msg);
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn flush(&self) {}

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            if self.color {
                Self::log_color(record);
            } else {
                Self::log_plain(record);
            }
        }
    }
}
