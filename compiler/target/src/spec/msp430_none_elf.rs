use crate::spec::{LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult, Endianness};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "msp430-none-elf".to_string(),
        target_endian: Endianness::Little,
        target_pointer_width: 16,
        target_c_int_width: "16".to_string(),
        data_layout: "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16".to_string(),
        arch: "msp430".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            executables: true,

            // The LLVM backend currently can't generate object files. To
            // workaround this LLVM generates assembly files which then we feed
            // to gcc to get object files. For this reason we have a hard
            // dependency on this specific gcc.
            asm_args: vec!["-mcpu=msp430".to_string()],
            linker: Some("msp430-elf-gcc".to_string()),

            // There are no atomic CAS instructions available in the MSP430
            // instruction set, and the LLVM backend doesn't currently support
            // compiler fences so the Atomic* API is missing on this target.
            // When the LLVM backend gains support for compile fences uncomment
            // the `singlethread: true` line and set `max_atomic_width` to
            // `Some(16)`.
            max_atomic_width: Some(0),
            atomic_cas: false,
            // singlethread: true,

            // Because these devices have very little resources having an
            // unwinder is too onerous so we default to "abort" because the
            // "unwind" strategy is very rare.
            panic_strategy: PanicStrategy::Abort,

            // Similarly, one almost always never wants to use relocatable
            // code because of the extra costs it involves.
            relocation_model: "static".to_string(),

            // Right now we invoke an external assembler and this isn't
            // compatible with multiple codegen units, and plus we probably
            // don't want to invoke that many gcc instances.
            default_codegen_units: Some(1),

            // Since MSP430 doesn't meaningfully support faulting on illegal
            // instructions, LLVM generates a call to abort() function instead
            // of a trap instruction. Such calls are 4 bytes long, and that is
            // too much overhead for such small target.
            trap_unreachable: false,

            // See the thumb_base.rs file for an explanation of this value
            emit_debug_gdb_scripts: false,

            ..Default::default()
        },
    })
}
