use std::ffi::CString;

use crate::llvm::*;
use crate::Result;

pub struct ModuleBuilder<'ctx> {
    context: &'ctx Context,
    module: Module,
    target_data: TargetData,
}
impl<'ctx> ModuleBuilder<'ctx> {
    pub fn new(name: &str, context: &'ctx Context, target_machine: &TargetMachine) -> Result<Self> {
        let target_data = target_machine.get_target_data();
        let module = Module::create(name, context, target_machine.as_ref())?;
        Ok(Self {
            context,
            module,
            target_data,
        })
    }

    pub fn finish(self) -> Module {
        self.module
    }

    #[inline]
    pub fn get_void_type(&self) -> LLVMTypeRef {
        use llvm_sys::core::LLVMVoidTypeInContext;

        unsafe { LLVMVoidTypeInContext(self.context.as_ref()) }
    }

    #[inline]
    pub fn get_i8_type(&self) -> LLVMTypeRef {
        self.get_integer_type(8)
    }

    #[inline]
    pub fn get_i32_type(&self) -> LLVMTypeRef {
        self.get_integer_type(32)
    }

    #[inline]
    pub fn get_i64_type(&self) -> LLVMTypeRef {
        self.get_integer_type(64)
    }

    pub fn get_usize_type(&self) -> LLVMTypeRef {
        use llvm_sys::target::LLVMIntPtrTypeInContext;

        unsafe { LLVMIntPtrTypeInContext(self.context.as_ref(), self.target_data.as_ref()) }
    }

    #[inline]
    pub fn get_term_type(&self) -> LLVMTypeRef {
        self.get_usize_type()
    }

    pub fn get_integer_type(&self, width: usize) -> LLVMTypeRef {
        use llvm_sys::core::LLVMIntTypeInContext;

        unsafe { LLVMIntTypeInContext(self.context.as_ref(), width as libc::c_uint) }
    }

    pub fn get_array_type(&self, size: usize, ty: LLVMTypeRef) -> LLVMTypeRef {
        use llvm_sys::core::LLVMArrayType;

        unsafe { LLVMArrayType(ty, size as libc::c_uint) }
    }

    pub fn get_struct_type(&self, name: Option<&str>, field_types: &[LLVMTypeRef]) -> LLVMTypeRef {
        use llvm_sys::core::LLVMStructCreateNamed;
        use llvm_sys::core::LLVMStructSetBody;
        use llvm_sys::core::LLVMStructTypeInContext;

        if let Some(name) = name {
            let cstr = CString::new(name).unwrap();
            unsafe {
                let ty = LLVMStructCreateNamed(self.context.as_ref(), cstr.as_ptr());
                LLVMStructSetBody(
                    ty,
                    field_types.as_ptr() as *mut _,
                    field_types.len() as libc::c_uint,
                    /* packed= */ false as libc::c_int,
                );
                ty
            }
        } else {
            unsafe {
                LLVMStructTypeInContext(
                    self.context.as_ref(),
                    field_types.as_ptr() as *mut _,
                    field_types.len() as libc::c_uint,
                    /* packed= */ false as libc::c_int,
                )
            }
        }
    }

    pub fn get_function_type(
        &self,
        ret: LLVMTypeRef,
        params: &[LLVMTypeRef],
        variadic: bool,
    ) -> LLVMTypeRef {
        use llvm_sys::core::LLVMFunctionType;

        let params_ptr = params.as_ptr() as *mut _;
        unsafe { LLVMFunctionType(ret, params_ptr, params.len() as libc::c_uint, variadic as libc::c_int) }
    }

    pub fn get_erlang_function_type(&self, arity: usize) -> LLVMTypeRef {
        let term_type = self.get_term_type();
        let mut params = Vec::with_capacity(arity);
        for _ in 0..arity {
            params.push(term_type);
        }
        self.get_function_type(term_type, params.as_slice(), /* variadic */ false)
    }

    pub fn get_opaque_function_type(&self) -> LLVMTypeRef {
        let void_type = self.get_void_type();
        self.get_function_type(void_type, &[], /* variadic */ false)
    }

    pub fn get_pointer_type(&self, ty: LLVMTypeRef) -> LLVMTypeRef {
        use llvm_sys::core::LLVMPointerType;

        unsafe {
            LLVMPointerType(ty, /* address_space= */ 0 as libc::c_uint)
        }
    }

    pub fn build_constant_int(&self, ty: LLVMTypeRef, value: isize) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstInt;

        unsafe {
            LLVMConstInt(
                ty,
                value as libc::c_ulonglong,
                /* sign_extend= */ true as libc::c_int,
            )
        }
    }

    pub fn build_constant_uint(&self, ty: LLVMTypeRef, value: usize) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstInt;

        unsafe {
            LLVMConstInt(
                ty,
                value as libc::c_ulonglong,
                /* sign_extend= */ false as libc::c_int,
            )
        }
    }

    pub fn build_constant_string(&self, s: &str) -> LLVMValueRef {
        let cstr = CString::new(s).unwrap();
        self.build_constant_cstring(cstr)
    }

    pub fn build_constant_cstring(&self, s: CString) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstStringInContext;

        let len = s.as_bytes().len();
        unsafe {
            LLVMConstStringInContext(
                self.context.as_ref(),
                s.as_ptr(),
                len as libc::c_uint,
                /* dont_null_terminate= */ true as libc::c_int,
            )
        }
    }

    pub fn build_constant_array(&self, ty: LLVMTypeRef, values: &[LLVMValueRef]) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstArray;

        let len = values.len() as libc::c_uint;
        unsafe { LLVMConstArray(ty, values.as_ptr() as *mut _, len) }
    }

    pub fn build_constant_struct(&self, ty: LLVMTypeRef, fields: &[LLVMValueRef]) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstNamedStruct;
        use llvm_sys::core::LLVMConstStructInContext;
        use llvm_sys::core::LLVMGetStructName;

        let name = unsafe { LLVMGetStructName(ty) };
        if name.is_null() {
            unsafe {
                LLVMConstStructInContext(
                    self.context.as_ref(),
                    fields.as_ptr() as *mut _,
                    fields.len() as libc::c_uint,
                    /* packed= */ false as libc::c_int,
                )
            }
        } else {
            unsafe {
                LLVMConstNamedStruct(ty, fields.as_ptr() as *mut _, fields.len() as libc::c_uint)
            }
        }
    }

    pub fn add_function(&self, name: &CString, ty: LLVMTypeRef) -> LLVMValueRef {
        use llvm_sys::core::LLVMAddFunction;
       
        unsafe { LLVMAddFunction(self.module.as_ref(), name.as_ptr(), ty) }
    }

    pub fn add_constant(
        &self,
        ty: LLVMTypeRef,
        name: &str,
        initializer: Option<LLVMValueRef>,
    ) -> LLVMValueRef {
        use llvm_sys::core::LLVMSetGlobalConstant;

        let global = self.add_global(ty, name, initializer);
        unsafe {
            LLVMSetGlobalConstant(global, true as libc::c_int);
        }
        global
    }

    pub fn add_global(
        &self,
        ty: LLVMTypeRef,
        name: &str,
        initializer: Option<LLVMValueRef>,
    ) -> LLVMValueRef {
        use llvm_sys::core::LLVMAddGlobal;

        let cstr = CString::new(name).unwrap();
        let global = unsafe { LLVMAddGlobal(self.module.as_ref(), ty, cstr.as_ptr()) };
        if let Some(init) = initializer {
            self.set_initializer(global, init);
        }
        global
    }

    pub fn set_initializer(&self, global: LLVMValueRef, constant: LLVMValueRef) {
        use llvm_sys::core::LLVMSetInitializer;

        unsafe {
            LLVMSetInitializer(global, constant);
        }
    }

    pub fn set_linkage(&self, value: LLVMValueRef, linkage: Linkage) {
        use llvm_sys::core::LLVMSetLinkage;

        unsafe {
            LLVMSetLinkage(value, linkage.into());
        }
    }

    pub fn set_alignment(&self, value: LLVMValueRef, alignment: usize) {
        use llvm_sys::core::LLVMSetAlignment;

        unsafe {
            LLVMSetAlignment(value, alignment as libc::c_uint);
        }
    }

    pub fn build_pointer_cast(&self, value: LLVMValueRef, ty: LLVMTypeRef) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstPointerCast;

        unsafe { LLVMConstPointerCast(value, ty) }
    }


    pub fn build_const_inbounds_gep(&self, value: LLVMValueRef, indices: &[usize]) -> LLVMValueRef {
        use llvm_sys::core::LLVMConstInBoundsGEP;

        let i32_type = self.get_i32_type();
        let indices_values = indices
            .iter()
            .map(|i| self.build_constant_uint(i32_type, *i))
            .collect::<Vec<_>>();
        let num_indices = indices_values.len() as libc::c_uint;
        unsafe { LLVMConstInBoundsGEP(value, indices_values.as_ptr() as *mut _, num_indices) }
    }
}
