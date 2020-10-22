    .section .text.__lumen_dynamic_apply,"ax",@progbits
    .globl __lumen_dynamic_apply
    .p2align 4
    .type __lumen_dynamic_apply,@function
__lumen_dynamic_apply:
    .cfi_startproc
    .cfi_personality 155, DW.ref.rust_eh_personality
    .cfi_lsda 27, .L_dyn_call_lsda
    # Save the parent base pointer for when control returns to this call frame.
    # CFA directives will inform the unwinder to expect %rbp at the bottom of the
    # stack for this frame, so this should be the last value on the stack in the caller
    pushq    %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq     %rsp, %rbp
    .cfi_def_cfa_register %rbp

    # Reserve static stack space
    subq $40, %rsp

    # Callee-saved registers
    movq  %r15, 32(%rsp)
    movq  %r14, 24(%rsp)
    movq  %r13, 16(%rsp)
    movq  %r12, 8(%rsp)
    movq  %rbx, (%rsp)

    # Save stack pointer at beginning of spill region
    movq     %rsp, %r15

    # First, push argc into %r12
    movq     %rdx, %r12

    # If we have no arguments, jump straight to the call
    cmpq     $0, %r12
    cmoveq   %rdi, %r12
    je       .L_dyn_call_call

    # Since we have at least one argument, we need argc in %r13 too, and
    # subtract 1 from %r12 to get the max index
    movq     %rdx, %r13
    subq     $1, %r12
    # Then subtract 6 from %r13, giving us the number of arguments to spill
    subq     $6, %r13
    # If we have nothing to spill, skip it
    jbe      .L_dyn_call_spill_done

    # Next we actually spill the arguments
    # TODO: Need to make sure stack is 16-byte aligned before call
.L_dyn_call_spill:
        # Index into %rsi, using %r12, stride is 8 bytes
        pushq  (%rsi, %r12, 8)
        # We're pushing from right to left, so %r12 is our index, %r13 is our counter,
        # and we need to decrement both of them, but only %r13 is used to determine when
        # we're done, since we're indexing from the end
        decq   %r12
        decq   %r13
        jnz    .L_dyn_call_spill

.L_dyn_call_spill_done:

    # Since we're about to call another function, we need to move
    # rdi/rsi/rdx to r12-14 to avoid clobbering them
    movq     %rdi, %r12
    movq     %rsi, %r13
    movq     %rdx, %r14

    # Next up, push up to 6 arguments in to registers,
    # and we always have at least one argument
    movq     (%r13), %rdi

    # If we have two arguments, push the second argument
    cmpq     $2, %r14
    cmovaeq   8(%r13), %rsi

    # arg3, ..arg6
    cmp      $3, %r14
    cmovaeq  16(%r13), %rdx
    cmp      $4, %r14
    cmovaeq  24(%r13), %rcx
    cmp      $5, %r14
    cmovaeq  32(%r13), %r8
    cmp      $6, %r14
    cmovaeq  40(%r13), %r9

.L_dyn_call_call:
    # Save stack pointer before spill region
    pushq    %r15

    # We've set up the call, now we execute it!
    # The call will have set rax/rdx appropriately for our return
    callq    *%r12

.L_dyn_call_finish:
    # Restore stack pointer to before spill region
    popq %rsp

    # Restore callee-saved registers
    movq  32(%rsp), %r15
    movq  24(%rsp), %r14
    movq  16(%rsp), %r13
    movq  8(%rsp), %r12
    movq  (%rsp),  %rbx

    addq $40, %rsp

    .cfi_restore %rbx
    .cfi_restore %r12
    .cfi_restore %r13
    .cfi_restore %r14
    .cfi_restore %r15

    leaveq
    retq

.L_dyn_call_landing_pad:
    movq %rax, -16(%rbp)
    movl %edx, -8(%rbp)
    movq -16(%rbp), %rdi
    callq _Unwind_Resume@PLT
    ud2

.L_dyn_call_end:
    .size __lumen_dynamic_apply, .L_dyn_call_end-__lumen_dynamic_apply
    .cfi_endproc

    .section .gcc_except_table,"a",@progbits
    .p2align 2
.L_dyn_call_lsda:
    .byte 255
    .byte 255
    .byte 1
    .uleb128 .L_dyn_call_cst_end-.L_dyn_call_cst_begin
.L_dyn_call_cst_begin:
    # Start of IP range
    .uleb128 .L_dyn_call_call-__lumen_dynamic_apply
    # Length of IP range
    .uleb128 .L_dyn_call_finish-.L_dyn_call_call
    # Landing pad address
    .uleb128 .L_dyn_call_landing_pad-__lumen_dynamic_apply
    # Offset into action table
    .byte 0
    .uleb128 .L_dyn_call_finish-__lumen_dynamic_apply
    .uleb128 .L_dyn_call_end-.L_dyn_call_finish
    .byte 0
    .byte 0

.L_dyn_call_cst_end:
    .p2align 2
