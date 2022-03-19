use paste::paste;

/// Register all available conversion passes globally
///
/// NOTE: You should prefer registering individual passes, but this can be useful for dev/debugging
pub fn register_all_conversion_passes() {
    extern "C" {
        #[link_name = "mlirRegisterConversionPasses"]
        fn mlir_register_conversion_passes();
    }

    unsafe { mlir_register_conversion_passes() }
}

macro_rules! namespaced_pass_impl {
    ($namespace:ident, $name:ident, $mnemonic:ident) => {
        paste! {
            namespaced_pass_impl!($namespace, $name, $mnemonic, [<$name Pass>], [<mlirCreate $namespace $name>], [<mlirRegister $namespace $name>]);
        }
    };

    ($namespace:ident, $name:ident, $mnemonic:ident, $ty:ident, $create_name:ident, $register_name:ident) => {
        #[repr(transparent)]
        pub struct $ty(crate::pass::OwnedPass);
        impl crate::pass::Pass for $ty {
            #[inline(always)]
            fn base(&self) -> crate::pass::PassBase {
                self.0.base()
            }

            #[inline(always)]
            fn to_owned(self) -> crate::pass::OwnedPass {
                self.0
            }
        }
        impl $ty {
            /// Creates a new instance of this pass
            ///
            /// NOTE: You must register passes before using them, see `register`
            pub fn new() -> $ty {
                paste! {
                    unsafe { [<mlir_create_ $namespace _ $mnemonic>]() }
                }
            }

            /// Register this pass for use globally
            ///
            /// NOTE: This only needs to be invoked once on each pass type.
            pub fn register() {
                paste! {
                    unsafe { [<mlir_register_ $namespace _ $mnemonic>]() }
                }
            }
        }

        paste! {
            extern "C" {
                #[link_name = stringify!($create_name)]
                fn [<mlir_create_ $namespace _ $mnemonic>]() -> $ty;
                #[link_name = stringify!($register_name)]
                fn [<mlir_register_ $namespace _ $mnemonic>]();
            }
        }
    };
}

macro_rules! namespaced_module_pass_impl {
    ($namespace:ident, $name:ident, $mnemonic:ident) => {
        namespaced_pass_impl!($namespace, $name, $mnemonic);
        paste! {
            impl crate::pass::OpPass<crate::Module> for [<$name Pass>] {}
        }
    };
}

macro_rules! conversion_pass_impl {
    ($name:ident, $mnemonic:ident) => {
        namespaced_pass_impl!(Conversion, $name, $mnemonic);
    };
}

macro_rules! module_conversion_pass_impl {
    ($name:ident, $mnemonic:ident) => {
        namespaced_module_pass_impl!(Conversion, $name, $mnemonic);
    };
}

namespaced_module_pass_impl!(CIR, ConvertCIRToLLVM, convert_cir_to_llvm);

conversion_pass_impl!(ConvertAffineToStandard, convert_affine_to_standard);
conversion_pass_impl!(ConvertArithmeticToLLVM, convert_arithmetic_to_llvm);
module_conversion_pass_impl!(ConvertArithmeticToSPIRV, convert_arithmetic_to_spirv);
conversion_pass_impl!(ConvertArmNeon2dToIntr, convert_arm_neon_2d_to_intr);
module_conversion_pass_impl!(ConvertAsyncToLLVM, convert_async_to_llvm);
conversion_pass_impl!(
    ConvertBufferizationToMemRef,
    convert_bufferization_to_mem_ref
);
conversion_pass_impl!(ConvertComplexToLLVM, convert_complex_to_llvm);
conversion_pass_impl!(ConvertComplexToStandard, convert_complex_to_standard);
module_conversion_pass_impl!(ConvertControlFlowToLLVM, convert_control_flow_to_llvm);
module_conversion_pass_impl!(ConvertControlFlowToSPIRV, convert_control_flow_to_spirv);
module_conversion_pass_impl!(ConvertFuncToLLVM, convert_func_to_llvm);
module_conversion_pass_impl!(ConvertFuncToSPIRV, convert_func_to_spirv);
module_conversion_pass_impl!(ConvertLinalgToLLVM, convert_linalg_to_llvm);
module_conversion_pass_impl!(ConvertLinalgToSPIRV, convert_linalg_to_spirv);
module_conversion_pass_impl!(ConvertLinalgToStandard, convert_linalg_to_standard);
conversion_pass_impl!(ConvertMathToLLVM, convert_math_to_llvm);
module_conversion_pass_impl!(ConvertMathToLibm, convert_math_to_libm);
module_conversion_pass_impl!(ConvertMathToSPIRV, convert_math_to_spirv);
module_conversion_pass_impl!(ConvertMemRefToLLVM, convert_mem_ref_to_llvm);
module_conversion_pass_impl!(ConvertMemRefToSPIRV, convert_mem_ref_to_spirv);
module_conversion_pass_impl!(ConvertOpenACCToLLVM, convert_open_acc_to_llvm);
module_conversion_pass_impl!(ConvertOpenACCToSCF, convert_open_acc_to_scf);
module_conversion_pass_impl!(ConvertOpenMPToLLVM, convert_open_mp_to_llvm);
module_conversion_pass_impl!(ConvertPDLToPDLInterp, convert_pdl_to_pdl_interp);
module_conversion_pass_impl!(ConvertSCFToOpenMP, convert_scf_to_open_mp);
module_conversion_pass_impl!(ConvertSPIRVToLLVM, convert_spirv_to_llvm);
conversion_pass_impl!(ConvertShapeConstraints, convert_shape_constraints);
module_conversion_pass_impl!(ConvertShapeToStandard, convert_shape_to_standard);
module_conversion_pass_impl!(ConvertTensorToSPIRV, convert_tensor_to_spirv);
module_conversion_pass_impl!(ConvertVectorToLLVM, convert_vector_to_llvm);
module_conversion_pass_impl!(ConvertVectorToROCDL, convert_vector_to_rocdl);
conversion_pass_impl!(ConvertVectorToSCF, convert_vector_to_scf);
module_conversion_pass_impl!(ConvertVectorToSPIRV, convert_vector_to_spirv);
module_conversion_pass_impl!(LowerHostCodeToLLVM, lower_host_code_to_llvm);
conversion_pass_impl!(ReconcileUnrealizedCasts, reconcile_unrealized_casts);
conversion_pass_impl!(SCFToControlFlow, scf_to_control_flow);
module_conversion_pass_impl!(SCFToSPIRV, scf_to_spirv);
conversion_pass_impl!(TosaToLinalg, tosa_to_linalg);
conversion_pass_impl!(TosaToLinalgNamed, tosa_to_linalg_named);
conversion_pass_impl!(TosaToSCF, tosa_to_scf);
conversion_pass_impl!(TosaToStandard, tosa_to_standard);
