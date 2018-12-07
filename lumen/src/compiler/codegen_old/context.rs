use std::path::Path;
use std::sync::{Once, ONCE_INIT};

#[macro_use]
use super::*;
use super::target::*;
use super::module::Module;

// Used to ensure LLVM is only initialized once
static ONCE: Once = ONCE_INIT;

pub enum IntegerType {
    I8,
    I16,
    I32,
    I64,
    I128
}

pub struct Context {
    context: llvm::prelude::LLVMContextRef,
    builder: llvm::prelude::LLVMBuilderRef,
    target: Target,
    machine: TargetMachine,
    modules: Vec<Module>,
    module_refs: Vec<llvm::prelude::LLVMModuleRef>,
}
impl Context {
    pub fn new() -> Result<Context, CodeGenError> {
        Context::new_with_target(Target::default()?)
    }
    pub fn new_with_target(target: Target) -> Result<Context, CodeGenError> {
        // Initialize LLVM one time only
        let initialized = false;
        ONCE.call_once(|| {
            unsafe {
                llvm::target::LLVM_InitializeAllTargetInfos();
                llvm::target::LLVM_InitializeAllTargets();
                llvm::target::LLVM_InitializeAllTargetMCs();
                llvm::target::LLVM_InitializeAllAsmParsers();
                llvm::target::LLVM_InitializeAllAsmPrinters();
            }
        });
        if !initialized {
            return Err(CodeGenError::LLVMError("unable to initialize LLVM".to_string()))
        }
        // Create context
        let target = Target::default()?;
        let machine = TargetMachine::new(&target);
        unsafe {
            let context = llvm::core::LLVMContextCreate();
            let builder = llvm::core::LLVMCreateBuilderInContext(context);
            Ok(Context { context, builder, target, machine, modules: Vec::new(), module_refs: Vec::new() })
        }
    }

    pub fn add_module(&mut self, name: &str, module: ModuleDecl) -> Result<(), CodeGenError> {
        let module = Module::from_ast(name, module)?;
        let module_ref = self.parse_ir(module.to_string())?;
        self.modules.push(module);
        self.module_refs.push(module_ref);
        Ok(())
    }

    fn parse_ir(&self, ir: String) -> Result<llvm::prelude::LLVMModuleRef, CodeGenError> {
        // First, create an LLVM memory buffer to hold the IR
        let len = ir.len();
        let ir = CString::new(ir).expect("generated IR represents an invalid C string");
        let name = CString::new("module").unwrap();
        let buf = llvm::core::LLVMCreateMemoryBufferWithMemoryRange(ir.as_ptr(), len, name.as_ptr(), 0);
        if buf.is_null() {
            return Err(CodeGenError::LLVMError("could not create LLVM memory buffer to parse IR".to_string()));
        }
        // Then, parse the IR from the memory buffer
        let mut module: *mut llvm::prelude::LLVMModuleRef = std::ptr::null_mut();
        let mut err: *mut *mut libc::c_char = std::ptr::null_mut();
        unsafe {
            let result = llvm::ir_reader::LLVMParseIRInContext(self.context, buf, module, err);
            if result != 0 {
                let err = c_str_to_str!(*err);
                return Err(CodeGenError::LLVMError(String::from(err)));
            }
        }
        Ok(*module)
    }

    pub fn verify(&self) -> Result<(), CodeGenError> {
        for m in self.module_refs.iter() {
            self.verify_module(*m)?;
        }
        Ok(())
    }

    fn verify_module(&self, m: llvm::prelude::LLVMModuleRef) -> Result<(), CodeGenError> {
        use llvm::analysis::LLVMVerifierFailureAction;

        let mut err: *mut libc::c_char = std::ptr::null_mut();
        let result = unsafe {
            llvm::analysis::LLVMVerifyModule(m, LLVMVerifierFailureAction::LLVMReturnStatusAction, &mut err)
        };
        if result != 0 {
            let err = c_str_to_str!(err);
            return Err(CodeGenError::LLVMError(String::from(err)));
        }
        Ok(())
    }

    pub fn emit(&self, out_type: OutputType) -> Result<Vec<(String, String)>, CodeGenError> {
        let emitted = Vec::new();
        for (i, module) in self.module_refs.iter().enumerate() {
            let buf = self.machine.emit(*module, out_type)?;
            let name = self.modules[i].name;
            emitted.push((name.to_string(), buf.to_string()))
        }
        Ok(emitted)
    }

    pub fn emit_file(&self, file: &Path, out_type: OutputType) -> Result<(), CodeGenError> {
        let basedir = file.parent().expect("invalid file path, expected parent directory");
        for (i, module) in self.module_refs.iter().enumerate() {
            let buf = self.machine.emit(*module, out_type)?;
            let name = self.modules[i].name;
            let file = basedir.join(format!("{}.{}", name, out_type));
            buf.write_to_file(&file);
        }
        Ok(())
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            llvm::core::LLVMDisposeBuilder(self.builder);
            llvm::core::LLVMContextDispose(self.context);
        }
    }
}
