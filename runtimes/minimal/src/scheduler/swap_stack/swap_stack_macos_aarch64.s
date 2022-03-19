    .p2align 4
    .global ___lumen_swap_stack
___lumen_swap_stack:
    .cfi_startproc
    .cfi_personality 155, _rust_eh_personality
    .cfi_lsda 255
    ; At this point the following registers are bound:
    ;
    ;   x0 <- prev: *mut CalleeSavedRegisters
    ;   x1 <- new: *const CalleeSavedRegisters
    ;   x2 <- FIRST_SWAP (needs to be in a register because 64-bit constants can't be encoded in `cmp*` instructions directly, so need to use reg64, reg64 form.)
    ;

    ; Save the frame pointer and return address for when control returns to this call frame.
    ; CFA directives will inform the unwinder to expect the frame pointer at the bottom of the
    ; stack for this frame, so this should be the last value on the stack in the caller
    stp x29, x30, [sp, #-16]
    add sp, sp, #16

    ; We save the frame pointer and return address registers to scratch
    ; so that we can setup CFA directives if this is the first swap of the
    ; target process
    mov  x29, x9
    mov  x30, x10

    ; Save the callee-saved registers of `prev`
    mov  sp, x11
    stp  x11, x29, [x0]
    stp  x27, x28, [x0, #16]
    stp  x25, x26, [x0, #32]
    stp  x23, x24, [x0, #48]
    stp  x21, x22, [x0, #64]
    stp  x19, x20, [x0, #80]

    ; Restore the callee-saved registers of `new`
    ldp  x11, x29, [x0]
    mov  x11, sp
    ldp  x27, x28, [x0, #16]
    ldp  x25, x26, [x0, #32]
    ldp  x23, x24, [x0, #48]
    ldp  x21, x22, [x0, #64]
    ldp  x19, x20, [x0, #80]

    ; The value of all the callee-saved registers has changed, so we
    ; need to inform the unwinder of that fact before proceeding
    .cfi_restore sp
    .cfi_restore x29
    .cfi_restore x28
    .cfi_restore x27
    .cfi_restore x26
    .cfi_restore x25
    .cfi_restore x24
    .cfi_restore x23
    .cfi_restore x22
    .cfi_restore x21
    .cfi_restore x20
    .cfi_restore x19

    ; If this is the first time swapping to this process,
    ; we need to to perform some one-time initialization to
    ; link the stack to the original parent stack (i.e. the scheduler),
    ; which is important for the unwinder
    cmp  x20, x2
    b.ne L_resume

    ; Ensure we never perform initialization twice
    ldr  x20, #0
    ; Make sure we initialize the return address register from x10
    ; This address returns to the point in the scheduler where it invoked swap_stack
    mov x30, x10
    ; Store the following from the top of the stack down:
    ; * return address
    ; * original frame pointer
    ; * pointer to parent's frame pointer for the unwinder to restore
    sub sp, sp, #24
    stp x29, x30, [sp, #8]
    str x9, [sp]

    ; These CFI directives inform the unwinder of where it can expect
    ; to find the CFA relative to the frame pointer. This matches how we've laid out the stack.
    ;
    ; - The current frame pointer is now 16 bytes (2 words) above the stack pointer.
    ; - 16 bytes _down_ from the current frame pointer is the value from x10 that
    ; we pushed, containing the parent call frame's stack pointer.
    ;
    ; The first directive tells the unwinder that it can expect to find the
    ; CFA (call frame address) 16 bytes above w29. The second directive then
    ; tells the unwinder that it can find the return address 8 bytes down from the
    ; current frame pointer. The third directive is similar and tells the unwinder
    ; that it can find the previous frame pointer 16 bytes down from the current frame
    ; pointer. The unwinder will restore the frame pointer from that slot, and will
    ; then expect to find the previous CFA 16 bytes above that address, allowing it
    ; to walk back to the parent frame
    .cfi_def_cfa w29, 16
    .cfi_offset w30, -8
    .cfi_offset w29, -16

    ; Now that the frames are linked, we can call the entry point. For now, this
    ; is __lumen_trap_exceptions, which expects to receive two arguments: the function
    ; being wrapped by the exception handler, and the value of the closure environment,
    ; _if_ it is a closure being called, otherwise the value of that argument is Term::NONE
    mov x21, x0
    mov x19, x1

    ; We're all set to call the start entry function!
    ; NOTE: This call never truly returns, as the exception handler calls __lumen_builtin_exit
    ; with the return value of the 'real' entry function, or with an exception if one
    ; is caught. However, swap_stack _does_ return for all other swaps, just not the first.
    br x22

L_resume:
    ; We land here only on a context switch, and since the last switch _away_ from
    ; this process pushed the frame pointer/return address on to the stack, we restore
    ; those values and proceed
    ldp x29, x30, [sp]
    add sp, sp, #16

L_ret:
    ; At this point we will return back to where execution left off:
    ; For the 'root' (scheduler) process, this returns back into `swap_process`;
    ; for all other processes, this returns to the code which was executing when
    ; it yielded, the address of which we've just restored into the link register
    ; so we can simply branch to that address
    br x30
    .cfi_endproc
