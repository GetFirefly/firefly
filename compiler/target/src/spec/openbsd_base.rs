use crate::spec::{FramePointer, RelroLevel, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions {
        os: "openbsd".into(),
        dynamic_linking: true,
        executables: true,
        families: vec!["unix".into()],
        has_rpath: true,
        abi_return_struct_as_int: true,
        position_independent_executables: true,
        frame_pointer: FramePointer::Always, // FIXME 43575: should be MayOmit...
        relro_level: RelroLevel::Full,
        dwarf_version: Some(2),
        ..Default::default()
    }
}
