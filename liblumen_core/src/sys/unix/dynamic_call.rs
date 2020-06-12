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
#[inline(never)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn apply(f: DynamicCallee, argv: *const usize, argc: usize) -> usize {
    let ret: usize;
    llvm_asm!("
        # First, push argc into %r12 and %r13, subtract 1 to get max index
        movq     %rdx, %r12
        movq     %rdx, %r13
        subq     $$1, %r12
        # Then subtract 6 from %r13, giving us the number of arguments to spill
        subq     $$6, %r13
        # If we have nothing to spill, skip it
        jbe      __dyn_call_spill_done

        # Next we actually spill the arguments
        __dyn_call_spill:
          # Index into %rdi, using %r12, stride is 8 bytes
          pushq  (%rdi, %r12, 8)
          # We're pushing from right to left, so %r12 is our index, %r13 is our counter,
          # and we need to decrement both of them, but only %r13 is used to determine when
          # we're done, since we're indexing from the end
          decq   %r12
          decq   %r13
          jnz    __dyn_call_spill

        __dyn_call_spill_done:

        # Since we're about to call another function, we need to move
        # rdi/rsi/rdx to r12-14 to avoid clobbering them
        movq     %rdi, %r12
        movq     %rsi, %r13
        movq     %rdx, %r14

        # Next up, push up to 6 arguments in to registers,
        # and we always have at least one argument
        movq     (%r13), %rdi

        # If we have two arguments, push the second argument
        cmpq     $$1, %r14
        cmovaq   8(%r13), %rsi

        # arg3, ..arg6
        cmp      $$2, %r14
        cmovaq   16(%r13), %rdx
        cmp      $$3, %r14
        cmovaq   24(%r13), %rcx
        cmp      $$4, %r14
        cmovaq   32(%r13), %r8
        cmp      $$5, %r14
        cmovaq   40(%r13), %r9

        # We've set up the call, now we execute it!
        # The call will have set rax/rdx appropriately for our return
        callq    *%r12

        # Move our result into 'ret'
        movq     %rax, %rdi

        # The compiler takes care of moving %rdi into %rax, and
        # inserts a `popq %rbp; retq` for us.
        "
    // This gets set after the inner call
    : "={rdi}"(ret)
    // Following the SystemV ABI, the first three arguments should be in these
    // three registers, but we're going to be explicit to ensure things are more
    // clear
    : "{rdi}"(f), "{rsi}"(argv), "{rdx}"(argc)
    // Mark all registers as clobbered, except those we've explicitly preserved
    : "rax",   "rbx",   "rcx",   "rdx",   "rsi", /*"rdi",   "rbp",   "rsp",*/
      "r8",    "r9",    "r10",   "r11",   "r12",   "r13",   "r14",   "r15",
      "mm0",   "mm1",   "mm2",   "mm3",   "mm4",   "mm5",   "mm6",   "mm7",
      "xmm0",  "xmm1",  "xmm2",  "xmm3",  "xmm4",  "xmm5",  "xmm6",  "xmm7",
      "xmm8",  "xmm9",  "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
      "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23",
      "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",
      "cc", "dirflag", "fpsr", "flags", "memory"
    : "volatile", "alignstack"
    );
    ret
}

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

    extern "C" fn adder(x: usize, y: usize) -> usize {
        x + y
    }
}
