use paste::paste;

/// Register all available transforms globally
///
/// NOTE: You should prefer registering individual passes, but this can be useful for dev/debugging
pub fn register_all_transform_passes() {
    extern "C" {
        #[link_name = "mlirRegisterTransformsPasses"]
        fn mlir_register_transforms_passes();
    }

    unsafe { mlir_register_transforms_passes() }
}

macro_rules! transform_pass_impl {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            transform_pass_impl!($name, $mnemonic, [<$name Pass>], [<mlirCreateTransforms $name>], [<mlirRegisterTransforms $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $create_name:ident, $register_name:ident) => {
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
                    unsafe { [<mlir_create_transforms_ $mnemonic>]() }
                }
            }

            /// Register this pass for use globally
            ///
            /// NOTE: This only needs to be invoked once on each pass type.
            pub fn register() {
                paste! {
                    unsafe { [<mlir_register_transforms_ $mnemonic>]() }
                }
            }
        }

        paste! {
            extern "C" {
                #[link_name = stringify!($create_name)]
                fn [<mlir_create_transforms_ $mnemonic>]() -> $ty;
                #[link_name = stringify!($register_name)]
                fn [<mlir_register_transforms_ $mnemonic>]();
            }
        }
    };
}

transform_pass_impl!(CSE, cse);
transform_pass_impl!(Canonicalizer, canonicalizer);
transform_pass_impl!(ControlFlowSink, control_flow_sink);
transform_pass_impl!(Inliner, inliner);
transform_pass_impl!(LocationSnapshot, location_snapshot);
transform_pass_impl!(LoopInvariantCodeMotion, licm);
transform_pass_impl!(PrintOpStats, print_op_stats);
transform_pass_impl!(SCCP, sccp);
transform_pass_impl!(StripDebugInfo, strip_debug_info);
transform_pass_impl!(SymbolDCE, symbol_dce);
transform_pass_impl!(SymbolPrivatize, symbol_privatize);
transform_pass_impl!(ViewOpGraph, view_op_graph);
