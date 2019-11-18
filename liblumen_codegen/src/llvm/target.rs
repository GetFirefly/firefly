#![allow(unused)]
use inkwell::support::LLVMString;

pub use inkwell::targets::InitializationConfig;
pub use inkwell::targets::Target;
pub use inkwell::targets::TargetMachine;
pub use inkwell::targets::TargetData;

use super::enums::{OptimizationLevel, RelocMode, CodeModel};

/// Constructs an LLVM target for the current system
#[inline]
pub fn current() -> Target {
    from_triple(default_triple().as_str())
        .expect("unable to load target info for current system!")
}

/// Returns the current system's target triple string
#[inline]
pub fn default_triple() -> String {
    let default = TargetMachine::get_default_triple();
    default.to_string()
}

/// Returns the current system's CPU name string
#[inline]
pub fn host_cpu() -> String {
    let host_cpu = TargetMachine::get_host_cpu_name();
    host_cpu.to_string()
}

/// Returns the current system's CPU feature string
#[inline]
pub fn host_features() -> String {
    let host_features = TargetMachine::get_host_cpu_features();
    host_features.to_string()
}

/// Constructs an LLVM target from a target triple string
#[inline]
pub fn from_triple(triple: &str) -> Result<Target, LLVMString> {
    Target::from_triple(triple)
}

/// Constructs an LLVM target from a target name
#[inline]
pub fn from_name(name: &str) -> Option<Target> {
    Target::from_name(name)
}

/// Initializes all LLVM targets
#[inline]
pub fn initialize() {
    initialize_with_config(&InitializationConfig::default());
}

/// Initializes all LLVM targets with the given InitializationConfig
#[inline]
pub fn initialize_with_config(config: &InitializationConfig) {
    Target::initialize_all(config)
}

/// Initializes just the native target
#[inline]
pub fn initialize_native() -> Result<(), String> {
    initialize_native_with_config(&InitializationConfig::default())
}

/// Initializes the native target with the given InitializationConfig
#[inline]
pub fn initialize_native_with_config(config: &InitializationConfig) -> Result<(), String> {
    Target::initialize_native(config)
}

/// Initializes the WebAssembly target
#[inline]
pub fn initialize_webassembly() {
    initialize_webassembly_with_config(&InitializationConfig::default());
}

/// Initializes the native target with the given InitializationConfig
#[inline]
pub fn initialize_webassembly_with_config(config: &InitializationConfig) {
    Target::initialize_webassembly(config)
}

/// Creates an LLVM target machine for the given triple+cpu+features,
/// and with the given optimization level, relocation mode, and code model.
/// 
/// If the target machine is unable to be created, returns `None`
#[inline]
pub fn create_target_machine(
    triple: &str, 
    cpu: &str, 
    features: &str, 
    opt: OptimizationLevel, 
    reloc: RelocMode, 
    model: CodeModel
) -> Option<TargetMachine> {
    let current = current();
    current.create_target_machine(triple, cpu, features, opt, reloc, model)
}