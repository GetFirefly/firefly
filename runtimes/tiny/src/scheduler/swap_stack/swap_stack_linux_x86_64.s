    .section .text.__firefly_swap_stack,"ax",@progbits
    .globl __firefly_swap_stack
    .p2align 4
    .type __firefly_swap_stack,@function
__firefly_swap_stack:
    .cfi_startproc
    .cfi_personality 155, DW.ref.rust_eh_personality
    .cfi_lsda 255
    # At this point the following registers are bound:
    #
    #   rdi <- prev: *mut CalleeSavedRegisters
    #   rsi <- new: *const CalleeSavedRegisters
    #   rdx <- FIRST_SWAP (needs to be in a register because 64-bit constants can't be encoded in `cmp*` instructions directly, so need to use reg64, reg64 form.)
    #

    # Save the return address to a register
    lea   rax, [rip + .L_ret]

    # Save the parent base pointer for when control returns to this call frame.
    # CFA directives will inform the unwinder to expect %rbp at the bottom of the
    # stack for this frame, so this should be the last value on the stack in the caller
    push  rbp

    # We also save %rbp and %rsp to registers so that we can setup CFA directives if this
    # is the first swap for the target process
    mov  rcx, rbp
    mov  r9, rsp

    # Save the stack pointer, and callee-saved registers of `prev`
    mov  [rdi], rsp
    mov  [rdi + 8], r15
    mov  [rdi + 16], r14
    mov  [rdi + 24], r13
    mov  [rdi + 32], r12
    mov  [rdi + 40], rbx
    mov  [rdi + 48], rbp

    # Restore the stack pointer, and callee-saved registers of `new`
    mov  rsp, [rsi]
    mov  r15, [rsi + 8]
    mov  r14, [rsi + 16]
    mov  r13, [rsi + 24]
    mov  r12, [rsi + 32]
    mov  rbx, [rsi + 40]
    mov  rbp, [rsi + 48]

    # The value of all the callee-saved registers has changed, so we
    # need to inform the unwinder of that fact before proceeding
    .cfi_restore rsp
    .cfi_restore r15
    .cfi_restore r14
    .cfi_restore r13
    .cfi_restore r12
    .cfi_restore rbx
    .cfi_restore rbp

    # If this is the first time swapping to this process,
    # we need to to perform some one-time initialization to
    # link the stack to the original parent stack (i.e. the scheduler),
    # which is important for the unwinder
    cmp  rdx, r13
    jne  .L_resume

    # Ensure we never perform initialization twice
    mov  r13, 0x0
    # Store the original base pointer at the top of the stack
    # And set the current base pointer to %rsp so that the unwinder
    # knows where to find the caller base pointer
    push rcx
    # Store the current frame pointer
    push rbp
    mov rbp, rsp

    # These CFI directives inform the unwinder of where it can expect
    # to find the CFA relative to %rbp. This matches how we've laid out the stack.
    #
    # - The current %rbp is now 24 bytes (3 words) above %rsp.
    # - 16 bytes _down_ from the current %rbp is the value from %r9 that
    # we pushed, containing the parent call frame's stack pointer.
    #
    # The first directive tells the unwinder that it can expect to find the
    # CFA (call frame address) 8 bytes above %rbp. The second directive then
    # tells the unwinder that it can find the previous %rbp 16 bytes _down_
    # from the current %rbp. The result is that the unwinder will restore %rbp
    # from that stack slot, and will then expect to find the previous CFA 16 bytes
    # above that address, allowing the unwinder to walk back into the parent frame
    .cfi_def_cfa rbp, 16
    .cfi_offset rbp, -16

    # Now that the frames are linked, we can call the entry point.
    # The only argument is the value of the closure environment (or Term::NONE if not a closure)
    mov  rdi, r12
    call r14

    # When we return to this point, the process has fully unwound and should exit, returning
    # back to the scheduler. We handle this by calling __firefly_builtin_exit, which sets up the
    # process status, and then yields to the scheduler. Control never returns here, so we hint
    # as such by which branch instruction we use
    #
    # NOTE: The ErlangResult struct will have been saved in rax/rdx, so we must move it to rdi/rsi
    # to reflect passing it by value as the sole argument to the __firefly_builtin_exit intrinsic
    mov rdi, rax
    mov rsi, rdx
    jmp __firefly_builtin_exit

.L_resume:
    # We land here only on a context switch, and since the last switch _away_ from
    # this process pushed %rbp on to the stack, and we don't need that value, we
    # adjust the stack pointer accordingly.
    add rsp, 8
    .cfi_def_cfa rsp, 8

.L_ret:
    # At this point we will return back to where execution left off:
    # For the 'root' (scheduler) process, this returns back into `swap_process`;
    # for all other processes, this returns to the code which was executing when
    # it yielded, the address of which is 8 bytes above the current stack pointer.
    # We pop and jmp rather than ret to avoid branch mispredictions.
    pop rax
    jmp rax

    .cfi_endproc
