use super::*;
use crate::support::*;

/// Represents a function in LLVM IR
///
/// Functions are subtypes of llvm::Value, llvm::GlobalValue, and llvm::GlobalObject
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Function(ValueBase);
impl Constant for Function {}
impl Value for Function {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl GlobalValue for Function {}
impl GlobalObject for Function {}
impl Function {
    pub fn arity(self) -> usize {
        extern "C" {
            fn LLVMCountParams(fun: Function) -> u32;
        }
        unsafe { LLVMCountParams(self) as usize }
    }

    pub fn arguments(self) -> Vec<ArgumentValue> {
        extern "C" {
            fn LLVMGetParams(fun: Function, args: *mut ArgumentValue);
        }
        let len = self.arity();
        let mut args = Vec::with_capacity(len);
        unsafe {
            LLVMGetParams(self, args.as_mut_ptr());
            args.set_len(len);
        }
        args
    }

    pub fn delete(self) {
        extern "C" {
            fn LLVMDeleteFunction(fun: Function);
        }
        unsafe { LLVMDeleteFunction(self) }
    }

    pub fn has_personality_fn(self) -> bool {
        extern "C" {
            fn LLVMHasPersonalityFn(fun: Function) -> bool;
        }
        unsafe { LLVMHasPersonalityFn(self) }
    }

    pub fn personality_fn(self) -> Function {
        extern "C" {
            fn LLVMGetPersonalityFn(fun: Function) -> Function;
        }
        unsafe { LLVMGetPersonalityFn(self) }
    }

    pub fn set_personality_fn(self, personality: Function) {
        extern "C" {
            fn LLVMSetPersonalityFn(fun: Function, personality: Function);
        }
        unsafe { LLVMSetPersonalityFn(self, personality) }
    }

    /// Get this function's calling convention
    pub fn calling_convention(self) -> CallConv {
        extern "C" {
            fn LLVMGetFunctionCallConv(fun: Function) -> u32;
        }
        unsafe { LLVMGetFunctionCallConv(self) }.into()
    }

    /// Set this function's calling convention
    pub fn set_calling_convention(self, cc: CallConv) {
        extern "C" {
            fn LLVMSetFunctionCallConv(fun: Function, cc: u32);
        }
        unsafe { LLVMSetFunctionCallConv(self, cc.into()) }
    }

    /// Get the name of the garbage collector strategy to use during codegen, if set
    pub fn gc(self) -> Option<StringRef> {
        extern "C" {
            fn LLVMGetGC(fun: Function) -> *const std::os::raw::c_char;
        }
        unsafe {
            let ptr = LLVMGetGC(self);
            if ptr.is_null() {
                None
            } else {
                Some(StringRef::from_ptr(ptr))
            }
        }
    }

    /// Set the name of the garbage collector strategy for this function
    pub fn set_gc<S: Into<StringRef>>(self, gc: S) {
        extern "C" {
            fn LLVMSetGC(fun: Function, name: *const std::os::raw::c_char);
        }
        let gc = gc.into();
        let c_str = gc.to_cstr();
        unsafe { LLVMSetGC(self, c_str.as_ptr()) }
    }

    pub fn num_blocks(self) -> usize {
        extern "C" {
            fn LLVMCountBasicBlocks(fun: Function) -> u32;
        }
        unsafe { LLVMCountBasicBlocks(self) as usize }
    }

    pub fn entry(self) -> Option<Block> {
        extern "C" {
            fn LLVMGetEntryBasicBlock(fun: Function) -> Block;
        }
        let block = unsafe { LLVMGetEntryBasicBlock(self) };
        if block.is_null() {
            None
        } else {
            Some(block)
        }
    }

    pub fn blocks(self) -> impl Iterator<Item = Block> {
        BlockIter::new(self)
    }

    pub fn append_block(self, block: Block) {
        extern "C" {
            fn LLVMAppendExistingBasicBlock(fun: Function, bb: Block);
        }
        unsafe { LLVMAppendExistingBasicBlock(self, block) }
    }

    /// Set an attribute on this function at the given index
    pub fn add_attribute_at_index<A: Attribute>(self, attr: A, index: AttributePlace) {
        attr.add(self, index)
    }

    /// Adds `attr` to this function
    pub fn add_attribute<A: Attribute>(self, attr: A) {
        self.add_attribute_at_index(attr, AttributePlace::Function);
    }

    /// Adds `attr` to the `n`th parameter of this function
    pub fn add_param_attribute<A: Attribute>(self, n: u32, attr: A) {
        self.add_attribute_at_index(attr, AttributePlace::Argument(n))
    }

    /// Adds `attr` to the return value of this function
    pub fn add_return_attribute<A: Attribute>(self, attr: A) {
        self.add_attribute_at_index(attr, AttributePlace::ReturnValue)
    }

    /// Removes `attr` from this function at the given index
    pub fn remove_attribute_at_index<A: Attribute>(self, attr: A, index: AttributePlace) {
        attr.remove(self, index)
    }

    /// Removes `attr` from the function
    pub fn remove_attribute<A: Attribute>(self, attr: A) {
        self.remove_attribute_at_index(attr, AttributePlace::Function)
    }

    /// Removes `attr` from the `n`th parameter of this function
    pub fn remove_param_attribute<A: Attribute>(self, n: u32, attr: A) {
        self.remove_attribute_at_index(attr, AttributePlace::Argument(n))
    }

    /// Removes `attr` from the return value of this function
    pub fn remove_return_attribute<A: Attribute>(self, attr: A) {
        self.remove_attribute_at_index(attr, AttributePlace::ReturnValue)
    }

    /// Set the debug info subprogram attached to this function
    pub fn set_di_subprogram(self, subprogram: Metadata) {
        extern "C" {
            fn LLVMSetSubprogram(fun: Function, subprogram: Metadata);
        }
        unsafe { LLVMSetSubprogram(self, subprogram) }
    }
}
impl TryFrom<ValueBase> for Function {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Function => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ArgumentValue(ValueBase);
impl Value for ArgumentValue {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ArgumentValue {
    pub fn parent(self) -> Function {
        extern "C" {
            fn LLVMGetParamParent(arg: ArgumentValue) -> Function;
        }
        unsafe { LLVMGetParamParent(self) }
    }

    pub fn set_alignment(self, align: usize) {
        extern "C" {
            fn LLVMSetParamAlignment(arg: ArgumentValue, align: u32);
        }
        unsafe { LLVMSetParamAlignment(self, align.try_into().unwrap()) }
    }
}
impl TryFrom<ValueBase> for ArgumentValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Argument => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum CallConv {
    C = 0,
    Fast = 8,
    Cold = 9,
    GHC = 10,
    HiPE = 11,
    WebkitJS = 12,
    AnyReg = 13,
    PreserveMost = 14,
    PreserveAll = 15,
    Swift = 16,
    CXX_FastTLS = 17,
    X86_Stdcall = 64,
    X86_Fastcall = 65,
    ARMAPCS = 66,
    ARMAAPCS = 67,
    ARMAAPCSVFP = 68,
    MSP430INTR = 69,
    X86_ThisCall = 70,
    PTXKernel = 71,
    PTXDevice = 72,
    SPIRFUNC = 75,
    SPIRKERNEL = 76,
    IntelOCLBI = 77,
    X8664SysV = 78,
    Win64 = 79,
    X86_VectorCall = 80,
    HHVM = 81,
    HHVMC = 82,
    X86INTR = 83,
    AVRINTR = 84,
    AVRSIGNAL = 85,
    AVRBUILTIN = 86,
    AMDGPUVS = 87,
    AMDGPUGS = 88,
    AMDGPUPS = 89,
    AMDGPUCS = 90,
    AMDGPUKERNEL = 91,
    X86_RegCall = 92,
    AMDGPUHS = 93,
    MSP430BUILTIN = 94,
    AMDGPULS = 95,
    AMDGPUES = 96,
    Other(u32),
}
impl Into<u32> for CallConv {
    fn into(self) -> u32 {
        match self {
            Self::C => 0,
            Self::Fast => 8,
            Self::Cold => 9,
            Self::GHC => 10,
            Self::HiPE => 11,
            Self::WebkitJS => 12,
            Self::AnyReg => 13,
            Self::PreserveMost => 14,
            Self::PreserveAll => 15,
            Self::Swift => 16,
            Self::CXX_FastTLS => 17,
            Self::X86_Stdcall => 64,
            Self::X86_Fastcall => 65,
            Self::ARMAPCS => 66,
            Self::ARMAAPCS => 67,
            Self::ARMAAPCSVFP => 68,
            Self::MSP430INTR => 69,
            Self::X86_ThisCall => 70,
            Self::PTXKernel => 71,
            Self::PTXDevice => 72,
            Self::SPIRFUNC => 75,
            Self::SPIRKERNEL => 76,
            Self::IntelOCLBI => 77,
            Self::X8664SysV => 78,
            Self::Win64 => 79,
            Self::X86_VectorCall => 80,
            Self::HHVM => 81,
            Self::HHVMC => 82,
            Self::X86INTR => 83,
            Self::AVRINTR => 84,
            Self::AVRSIGNAL => 85,
            Self::AVRBUILTIN => 86,
            Self::AMDGPUVS => 87,
            Self::AMDGPUGS => 88,
            Self::AMDGPUPS => 89,
            Self::AMDGPUCS => 90,
            Self::AMDGPUKERNEL => 91,
            Self::X86_RegCall => 92,
            Self::AMDGPUHS => 93,
            Self::MSP430BUILTIN => 94,
            Self::AMDGPULS => 95,
            Self::AMDGPUES => 96,
            Self::Other(cc) => cc,
        }
    }
}
impl From<u32> for CallConv {
    fn from(cc: u32) -> Self {
        match cc {
            0 => Self::C,
            8 => Self::Fast,
            9 => Self::Cold,
            10 => Self::GHC,
            11 => Self::HiPE,
            12 => Self::WebkitJS,
            13 => Self::AnyReg,
            14 => Self::PreserveMost,
            15 => Self::PreserveAll,
            16 => Self::Swift,
            17 => Self::CXX_FastTLS,
            64 => Self::X86_Stdcall,
            65 => Self::X86_Fastcall,
            66 => Self::ARMAPCS,
            67 => Self::ARMAAPCS,
            68 => Self::ARMAAPCSVFP,
            69 => Self::MSP430INTR,
            70 => Self::X86_ThisCall,
            71 => Self::PTXKernel,
            72 => Self::PTXDevice,
            75 => Self::SPIRFUNC,
            76 => Self::SPIRKERNEL,
            77 => Self::IntelOCLBI,
            78 => Self::X8664SysV,
            79 => Self::Win64,
            80 => Self::X86_VectorCall,
            81 => Self::HHVM,
            82 => Self::HHVMC,
            83 => Self::X86INTR,
            84 => Self::AVRINTR,
            85 => Self::AVRSIGNAL,
            86 => Self::AVRBUILTIN,
            87 => Self::AMDGPUVS,
            88 => Self::AMDGPUGS,
            89 => Self::AMDGPUPS,
            90 => Self::AMDGPUCS,
            91 => Self::AMDGPUKERNEL,
            92 => Self::X86_RegCall,
            93 => Self::AMDGPUHS,
            94 => Self::MSP430BUILTIN,
            95 => Self::AMDGPULS,
            96 => Self::AMDGPUES,
            n => Self::Other(n),
        }
    }
}
