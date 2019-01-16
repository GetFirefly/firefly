use std::ffi::CStr;

use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::*;
use llvm_sys::transforms::pass_manager_builder;

use super::memory_buffer::MemoryBuffer;
use super::*;

pub struct TargetMachine {
    machine: LLVMTargetMachineRef,
    pm: LLVMPassManagerRef,
    layout: LLVMTargetDataRef,
}
impl TargetMachine {
    pub fn new(target: &Target) -> TargetMachine {
        let level = Optimization::Default;
        let cpu = c_str!("generic");
        let features = c_str!("");
        let opt_level = LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault;
        let reloc_mode = LLVMRelocMode::LLVMRelocDefault;
        let code_model = LLVMCodeModel::LLVMCodeModelDefault;
        let machine = unsafe {
            LLVMCreateTargetMachine(
                target.target,
                target.c_str(),
                cpu,
                features,
                opt_level,
                reloc_mode,
                code_model,
            )
        };

        // Map optimization level to LLVM type
        let level: LLVMCodeGenOptLevel = Optimization::into(level);

        // Set analysis passes for this target machine
        let internalize = 0;
        let inline = 0;
        let pm = unsafe { LLVMCreatePassManager() };
        unsafe {
            let pmb = pass_manager_builder::LLVMPassManagerBuilderCreate();
            pass_manager_builder::LLVMPassManagerBuilderSetOptLevel(pmb, level as u32);
            pass_manager_builder::LLVMPassManagerBuilderPopulateLTOPassManager(
                pmb,
                pm,
                internalize,
                inline,
            );
            pass_manager_builder::LLVMPassManagerBuilderDispose(pmb);
            LLVMAddAnalysisPasses(machine, pm);
        }
        let layout = unsafe { LLVMCreateTargetDataLayout(machine) };
        TargetMachine {
            machine,
            pm,
            layout,
        }
    }

    pub fn emit_ir(&self, module: LLVMModuleRef) -> Result<MemoryBuffer, CodeGenError> {
        let out = unsafe { LLVMPrintModuleToString(module) };
        Ok(MemoryBuffer::from_ptr(out))
    }

    pub fn emit_asm(&self, module: LLVMModuleRef) -> Result<MemoryBuffer, CodeGenError> {
        self.emit(module, OutputType::Assembly)
    }

    pub fn emit_obj(&self, module: LLVMModuleRef) -> Result<MemoryBuffer, CodeGenError> {
        self.emit(module, OutputType::Object)
    }

    pub fn emit(
        &self,
        module: LLVMModuleRef,
        output_type: OutputType,
    ) -> Result<MemoryBuffer, CodeGenError> {
        let output_type = match output_type {
            OutputType::IR => return self.emit_ir(module),
            OutputType::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputType::Object => LLVMCodeGenFileType::LLVMObjectFile,
        };
        let out: *mut LLVMMemoryBufferRef = std::ptr::null_mut();
        let err = std::ptr::null_mut();
        let result = unsafe {
            LLVMTargetMachineEmitToMemoryBuffer(self.machine, module, output_type, err, out)
        };
        if result != 0 {
            let err = unsafe {
                CStr::from_ptr(err as *mut libc::c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(CodeGenError::LLVMError(format!(
                "failed to emit IR: {}",
                err
            )));
        }
        Ok(MemoryBuffer::from_ref(unsafe { *out }))
    }
}
impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposePassManager(self.pm);
            LLVMDisposeTargetData(self.layout);
            LLVMDisposeTargetMachine(self.machine);
        }
    }
}

pub struct Target {
    name: String,
    target: LLVMTargetRef,
}
impl Target {
    pub fn default() -> Result<Target, CodeGenError> {
        let default = unsafe { LLVMGetDefaultTargetTriple() };
        Target::from_triple(c_str_to_str!(default))
    }

    pub fn from_triple(triple: &str) -> Result<Target, CodeGenError> {
        let target = unsafe {
            let mut tref: LLVMTargetRef = std::ptr::null_mut();
            let err: *mut *mut libc::c_char = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(c_str!(triple), &mut tref, err) != 0 {
                let reason = String::from(c_str_to_str!(*err));
                return Err(CodeGenError::LLVMError(reason));
            }
            tref
        };
        Ok(Target {
            name: triple.to_string(),
            target,
        })
    }

    pub fn c_str(&self) -> *const libc::c_char {
        c_str!(self.name.as_str())
    }
}
