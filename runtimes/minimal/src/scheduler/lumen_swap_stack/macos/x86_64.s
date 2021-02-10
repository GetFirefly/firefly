    .p2align 4
    .global ___lumen_swap_stack
___lumen_swap_stack:
    # Save the return address to a register
    leaq    0f(%rip),   %rax

    # Save the parent base pointer for when control returns to this call frame.
    # CFA directives will inform the unwinder to expect %rbp at the bottom of the
    # stack for this frame, so this should be the last value on the stack in the caller
    pushq    %rbp

    # We also save %rbp and %rsp to registers so that we can setup CFA directives if this
    # is the first swap for the target process
    movq     %rbp, %rcx
    movq     %rsp, %r9

    # Save the stack pointer, and callee-saved registers of `prev`
    movq     %rsp, (%rdi)
    movq     %r15, 8(%rdi)
    movq     %r14, 16(%rdi)
    movq     %r13, 24(%rdi)
    movq     %r12, 32(%rdi)
    movq     %rbx, 40(%rdi)
    movq     %rbp, 48(%rdi)

    # Restore the stack pointer, and callee-saved registers of `new`
    movq     (%rsi),   %rsp
    movq     8(%rsi),  %r15
    movq     16(%rsi), %r14
    movq     24(%rsi), %r13
    movq     32(%rsi), %r12
    movq     40(%rsi), %rbx
    movq     48(%rsi), %rbp

    # The value of all the callee-saved registers has changed, so we
    # need to inform the unwinder of that fact before proceeding
    .cfi_restore %rsp
    .cfi_restore %r15
    .cfi_restore %r14
    .cfi_restore %r13
    .cfi_restore %r12
    .cfi_restore %rbx
    .cfi_restore %rbp

    # If this is the first time swapping to this process,
    # we need to to perform some one-time initialization to
    # link the stack to the original parent stack (i.e. the scheduler),
    # which is important for the unwinder
    cmpq     %r13, %rdx
    jne      L_resume

    # Ensure we never perform initialization twice
    movq  $$0x0, %r13
    # Store the original base pointer at the top of the stack
    pushq %rcx
    # Followed by the return address
    pushq %rax
    # Finally we store a pointer to the bottom of the stack in the
    # parent call frame. The unwinder will expect to restore %rbp
    # from this address
    pushq %r9

    # These CFI directives inform the unwinder of where it can expect
    # to find the CFA relative to %rbp. This matches how we've laid out the stack.
    #
    # - The current %rbp is now 24 bytes (3 words) above %rsp.
    # - 16 bytes _down_ from the current %rbp is the value from %r9 that
    # we pushed, containing the parent call frame's stack pointer.
    #
    # The first directive tells the unwinder that it can expect to find the
    # CFA (call frame address) 16 bytes above %rbp. The second directive then
    # tells the unwinder that it can find the previous %rbp 16 bytes _down_
    # from the current %rbp. The result is that the unwinder will restore %rbp
    # from that stack slot, and will then expect to find the previous CFA 16 bytes
    # above that address, allowing the unwinder to walk back into the parent frame
    .cfi_def_cfa %rbp, 16
    .cfi_offset %rbp, -16

    # Now that the frames are linked, we can call the entry point. For now, this
    # is __lumen_trap_exceptions, which expects to receive two arguments: the function
    # being wrapped by the exception handler, and the value of the closure environment,
    # _if_ it is a closure being called, otherwise the value of that argument is Term::NONE
    movq  %r14, %rdi
    movq  %r12, %rsi

    # We have already set up the stack precisely, so we don't use callq here, instead
    # we go ahead and jump straight to the beginning of the entry function.
    # NOTE: This call never truly returns, as the exception handler calls __lumen_builtin_exit
    # with the return value of the 'real' entry function, or with an exception if one
    # is caught. However, swap_stack _does_ return for all other swaps, just not the first.
    jmpq *%r15

L_resume:
    # We land here only on a context switch, and since the last switch _away_ from
    # this process pushed %rbp on to the stack, and we don't need that value, we
    # adjust the stack pointer accordingly.
    add $$8, %rsp

    # At this point we will return back to where execution left off:
    # For the 'root' (scheduler) process, this returns back into `swap_process`;
    # for all other processes, this returns to the code which was executing when
    # it yielded, the address of which is 8 bytes above the current stack pointer.
    # We pop and jmp rather than ret to avoid branch mispredictions.
    popq %rax
    jmpq *%rax
