///! This module defines an assembly shim that dispatches calls with a
///! variable number of arguments as efficiently as possible, following the
///! System-V ABI calling convention, or other equivalent platform-standard
///! convention, depending on target.
///!
///! Currently, we have written the shim for x86_64 Linux and macOS.
///!
///! See the assembly files in `dynamic_apply/*.s` for details on their
///! implementation.
use core::arch::global_asm;

use cfg_if::cfg_if;

use crate::function::ErlangResult;
#[cfg(feature = "async")]
use crate::futures::ErlangFuture;
use crate::process::ProcessLock;
use crate::term::OpaqueTerm;

#[cfg(feature = "async")]
use super::DynamicAsyncCallee;
use super::DynamicCallee;

extern "C-unwind" {
    #[allow(improper_ctypes)]
    #[link_name = "__firefly_dynamic_apply"]
    pub fn apply(
        f: DynamicCallee,
        process: &mut ProcessLock,
        argv: *const OpaqueTerm,
        argc: usize,
    ) -> ErlangResult;

    #[cfg(feature = "async")]
    #[allow(improper_ctypes)]
    #[link_name = "__firefly_dynamic_apply_async"]
    pub fn apply_async(
        f: DynamicAsyncCallee,
        process: &mut ProcessLock,
        argv: *const OpaqueTerm,
        argc: usize,
    ) -> ErlangFuture;
}

cfg_if! {
    if #[cfg(all(target_os = "macos", target_arch = "x86_64"))] {
        global_asm!(include_str!("asm/dynamic_apply_macos.s"));
    } else if #[cfg(all(target_os = "macos", target_arch = "aarch64"))] {
        global_asm!(include_str!("asm/dynamic_apply_macos_aarch64.s"));
    } else if #[cfg(target_arch = "x86_64")] {
        global_asm!(include_str!("asm/dynamic_apply_linux.s"));
    } else {
        compile_error!("dynamic calls have not been implemented for this platform!");
    }
}

/*
#[cfg(test)]
mod tests {
    use core::mem;

    use super::*;
    use crate::term::OpaqueTerm;

    #[test]
    fn basic_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = adder as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[fixnum!(22), fixnum!(11)];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        let expected = ErlangResult::ok(fixnum!(33));
        assert_eq!(result, expected);
    }

    #[test]
    fn basic_apply_rustcc_test() {
        // Transform a function reference to a generic void function pointer
        let callee = adder_rust as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[fixnum!(22), fixnum!(11)];
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        let expected = ErlangResult::ok(fixnum!(33));
        assert_eq!(result, expected);
    }

    #[test]
    fn spilled_args_even_spills_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = spilled_args_even as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let mut args = vec![];
        args.resize(10, fixnum!(1));
        let (args, expected) = if cfg!(target_arch = "x86_64") {
            // On x86_64, we have 6 registers to use, so pass 8 arguments
            let slice = &args[0..7];
            (slice, fixnum!(8))
        } else if cfg!(target_arch = "aarch64") {
            // On aarch64, we have 8 registers to use, so pass 10 arguments
            (&args[0..], fixnum!(10))
        } else {
            panic!("need to update test case for this target");
        };
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        let expected = ErlangResult::ok(expected);
        assert_eq!(result, expected);
    }

    #[test]
    fn spilled_args_odd_spills_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = spilled_args_odd as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let mut args = vec![];
        args.resize(9, fixnum!(1));
        let (args, expected) = if cfg!(target_arch = "x86_64") {
            // On x86_64, we have 6 registers to use, so pass 7 arguments
            let slice = &args[0..6];
            (slice, fixnum!(7))
        } else if cfg!(target_arch = "aarch64") {
            // On aarch64, we have 8 registers to use, so pass 9 arguments
            (&args[0..], fixnum!(9))
        } else {
            panic!("need to update test case for this target");
        };
        let argv = args.as_ptr();
        let argc = args.len();
        let result = unsafe { apply(callee, argv, argc) };

        let expected = ErlangResult::ok(expected);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn panic_apply_test() {
        // Transform a function reference to a generic void function pointer
        let callee = panicky as *const ();
        // Transform the pointer to our DynamicCallee type alias, since that is what apply expects
        let callee = unsafe { mem::transmute::<*const (), DynamicCallee>(callee) };
        // Build up the args and call the function
        let args = &[fixnum!(22), fixnum!(11)];
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
        let mut args = vec![];
        args.resize(8, fixnum!(1));
        let argv = args.as_ptr();
        let argc = args.len();
        let _result = unsafe { apply(callee, argv, argc) };
    }

    fn panicky(_x: usize, _y: usize) -> ErlangResult {
        panic!("panicky");
    }

    extern "C-unwind" fn panicky_spilled(
        _a: Term,
        _b: Term,
        _c: Term,
        _d: Term,
        _e: Term,
        _f: Term,
        _g: Term,
    ) -> ErlangResult {
        panic!("panicky");
    }

    extern "C" fn adder(x: Term, y: Term) -> ErlangResult {
        let result = x.decode_immediate() + y.decode_immediate();
        ErlangResult::ok(fixnum!(result))
    }

    fn adder_rust(x: Term, y: Term) -> ErlangResult {
        let result = x.decode_immediate() + y.decode_immediate();
        ErlangResult::ok(fixnum!(result))
    }

    #[cfg(target_arch = "x86_64")]
    extern "C" fn spilled_args_even(
        a: Term,
        b: Term,
        c: Term,
        d: Term,
        e: Term,
        f: Term,
        g: Term,
        h: Term,
    ) -> ErlangResult {
        let a = a.decode_immediate();
        let b = b.decode_immediate();
        let c = c.decode_immediate();
        let d = d.decode_immediate();
        let e = e.decode_immediate();
        let f = f.decode_immediate();
        let g = g.decode_immediate();
        let h = h.decode_immediate();
        let value = fixnum!(a + b + c + d + e + f + g + h);
        ErlangResult::ok(value)
    }

    #[cfg(target_arch = "aarch64")]
    extern "C" fn spilled_args_even(
        a: Term,
        b: Term,
        c: Term,
        d: Term,
        e: Term,
        f: Term,
        g: Term,
        h: Term,
        i: Term,
        j: Term,
    ) -> ErlangResult {
        let a = a.decode_immediate();
        let b = b.decode_immediate();
        let c = c.decode_immediate();
        let d = d.decode_immediate();
        let e = e.decode_immediate();
        let f = f.decode_immediate();
        let g = g.decode_immediate();
        let h = h.decode_immediate();
        let i = i.decode_immediate();
        let j = j.decode_immediate();
        let value = fixnum!(a + b + c + d + e + f + g + h + i + j);
        ErlangResult::ok(value)
    }

    #[cfg(target_arch = "x86_64")]
    extern "C" fn spilled_args_odd(
        a: Term,
        b: Term,
        c: Term,
        d: Term,
        e: Term,
        f: Term,
        g: Term,
    ) -> ErlangResult {
        let a = a.decode_immediate();
        let b = b.decode_immediate();
        let c = c.decode_immediate();
        let d = d.decode_immediate();
        let e = e.decode_immediate();
        let f = f.decode_immediate();
        let g = g.decode_immediate();
        let value = fixnum!(a + b + c + d + e + f + g);
        ErlangResult::ok(value)
    }

    #[cfg(target_arch = "aarch64")]
    extern "C" fn spilled_args_odd(
        a: Term,
        b: Term,
        c: Term,
        d: Term,
        e: Term,
        f: Term,
        g: Term,
        h: Term,
        i: Term,
    ) -> ErlangResult {
        let a = a.decode_immediate();
        let b = b.decode_immediate();
        let c = c.decode_immediate();
        let d = d.decode_immediate();
        let e = e.decode_immediate();
        let f = f.decode_immediate();
        let g = g.decode_immediate();
        let h = h.decode_immediate();
        let i = i.decode_immediate();
        let value = fixnum!(a + b + c + d + e + f + g + h + i);
        ErlangResult::ok(value)
    }
}
*/
