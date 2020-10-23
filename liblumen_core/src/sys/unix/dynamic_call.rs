///! This module defines an assembly shim that dispatches calls with a
///! variable number of arguments as efficiently as possible, following the
///! System-V ABI calling convention, or other equivalent platform-standard
///! convention, depending on target.
///!
///! Currently, we have written the shim for x86_64 Linux and macOS.
///!
///! See the assembly files in `dynamic_apply/*.s` for details on their
///! implementation.
use crate::sys::dynamic_call::DynamicCallee;

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

    #[test]
    fn spilled_args_even_spills_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = spilled_args_even as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[1, 1, 1, 1, 1, 1, 1, 1, 1];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        assert_eq!(result, 8);
    }

    #[test]
    fn spilled_args_odd_spills_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = spilled_args_odd as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[1, 1, 1, 1, 1, 1, 1, 1];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        assert_eq!(result, 7);
    }

    #[test]
    #[should_panic]
    fn panic_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = panicky as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[22, 11];
        let argv = args.as_ptr();
        let argc = args.len();
        let _result = unsafe { apply(callee, argv, argc) };
    }

    #[test]
    #[should_panic]
    fn panic_apply_spills_test() {
        // Transform a function reference to a generic void function pointer
        let callee = panicky_spilled as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[1, 1, 1, 1, 1, 1, 1, 1];
        let argv = args.as_ptr();
        let argc = args.len();
        let _result = unsafe { apply(callee, argv, argc) };
    }

    fn panicky(_x: usize, _y: usize) -> usize {
        panic!("panicky");
    }

    #[unwind(allowed)]
    extern "C" fn panicky_spilled(
        _a: usize,
        _b: usize,
        _c: usize,
        _d: usize,
        _e: usize,
        _f: usize,
        _g: usize,
    ) -> usize {
        panic!("panicky");
    }

    extern "C" fn adder(x: usize, y: usize) -> usize {
        x + y
    }

    fn adder_rust(x: usize, y: usize) -> usize {
        x + y
    }

    extern "C" fn spilled_args_even(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        e: usize,
        f: usize,
        g: usize,
        h: usize,
    ) -> usize {
        a + b + c + d + e + f + g + h
    }

    extern "C" fn spilled_args_odd(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        e: usize,
        f: usize,
        g: usize,
    ) -> usize {
        a + b + c + d + e + f + g
    }
}
