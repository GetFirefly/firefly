use crate::spec::{LinkArgs, LinkerFlavor, RelroLevel, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            // GNU-style linkers will use this to omit linking to libraries
            // which don't actually fulfill any relocations, but only for
            // libraries which follow this flag.  Thus, use it before
            // specifying libraries to link to.
            "-Wl,--as-needed".to_string(),
        ],
    );

    TargetOptions {
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        no_default_libraries: false,
        has_rpath: true,
        pre_link_args: args,
        position_independent_executables: true,
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
