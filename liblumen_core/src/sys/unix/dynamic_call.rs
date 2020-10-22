use crate::sys::dynamic_call::DynamicCallee;

/// This function is horrific. The fact that it needs to exist could not
/// be more disappointing. But here we are. We need a way to dynamically
/// invoke functions in some specific situations, and this is our only option.
///
/// This was crafted following the SystemV ABI for x86_64, in short:
///
/// - Callee-saved registers are %rsp, %rbp, %rbx, %r12-15
/// - All other registers are caller-saved
/// - First six integer/pointer arguments are %rdi, %rsi, %rdx, %rcx, %r8, %r9
/// - If there are more than six arguments, they are spilled to the stack
///
/// This function needs to dynamically construct an ABI-correct call to
/// the provided function pointer. Working in our favor is the fact that
/// the argument and return types are the same, and are all integer-sized.
///
/// The gist:
///
/// - The compiler takes care of preparing the stack and saving/restoring callee-saved registers,
///   since this is not marked naked.
/// - Our first step is to determine if we need to allocate any spill storage
///   - If no spillage, skip this step
///   - If so, calculate the amount of spillage, and push spilled args to the stack, in reverse
///     order, i.e. the lowest index resides nearest the top of the stack
/// - Store up to 6 arguments from the provided argument vector in available parameter registers
/// - At this point, the call is prepared, so we execute it
/// - Write the return value to our local variable created to hold it
/// - Return to the caller
///
/// It is essential that this function never be inlined, or it would more than likely be
/// incorrect, so we tell the compiler not to do that.

extern "C" {
    #[unwind(allowed)]
    #[link_name = "__lumen_dynamic_apply"]
    pub fn apply(f: DynamicCallee, argv: *const usize, argc: usize) -> usize;
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
global_asm!(include_str!("dynamic_call/lumen_dynamic_apply_macos.s"));
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
global_asm!(include_str!("dynamic_call/lumen_dynamic_apply_linux.s"));

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn basic_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = adder as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[22, 11];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        assert_eq!(result, 33);
    }

    #[test]
    fn basic_apply_rustcc_test() {
        // Transform a function reference to a generic void function pointer
        let callee = adder_rust as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[22, 11];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        assert_eq!(result, 33);
    }

    extern "C" fn adder(x: usize, y: usize) -> usize {
        x + y
    }

    fn adder_rust(x: usize, y: usize) -> usize {
        x + y
    }
}
