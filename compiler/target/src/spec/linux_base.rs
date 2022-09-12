use std::borrow::Cow;

use crate::spec::{RelroLevel, SplitDebugInfo, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "linux".into(),
        dynamic_linking: true,
        families: vec!["unix".into()],
        has_rpath: true,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        has_thread_local: true,
        crt_static_respected: true,
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebugInfo::Packed,
            SplitDebugInfo::Unpacked,
            SplitDebugInfo::Off,
        ]),
        ..Default::default()
    }
}
