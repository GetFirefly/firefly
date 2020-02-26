use std::env;

use log::LevelFilter;

use anyhow::bail;

pub fn init(level: LevelFilter) -> anyhow::Result<()> {
    // Initialize logger
    let mut builder = env_logger::from_env("LUMEN_DEBUG_LOG");
    builder.format_indent(Some(2));
    if let Ok(precision) = env::var("LUMEN_LOG_WITH_TIME") {
        match precision.as_str() {
            "s" => builder.format_timestamp_secs(),
            "ms" => builder.format_timestamp_millis(),
            "us" => builder.format_timestamp_micros(),
            "ns" => builder.format_timestamp_nanos(),
            other => bail!(
                "invalid LUMEN_LOG_WITH_TIME precision, expected one of [s, ms, us, ns], got '{}'",
                other
            ),
        };
    } else {
        builder.format_timestamp(None);
    }
    builder.init();

    Ok(())
}
