    .p2align 2
    .globl ___firefly_dynamic_apply_async
___firefly_dynamic_apply_async:
L_dyn_call_begin:
    .cfi_startproc
    .cfi_personality 155, _rust_eh_personality
    .cfi_lsda 16, L_dyn_call_lsda
    ; At this point, the following registers are bound:
    ;
    ;   x29  <- frame pointer (optional)
    ;   x30  <- return address
    ;   x0   <- callee
    ;   x1   <- argv
    ;   x2   <- argc
    ;
    ; Save the parent base pointer for when control returns to this call frame.
    ; CFA directives will inform the unwinder to expect x29 at the bottom of the
    ; stack for this frame, so this should be the last value on the stack in the caller
    stp x29, x30, [sp, #-16]
    ; Set the frame pointer to the current stack pointer, we use this later to restore
    ; the original frame/stack pointers.
    mov x29, sp
    ; Define the call frame address relative to the frame pointer
    .cfi_def_cfa w29, 16
    .cfi_offset w30, -8
    .cfi_offset w29, -16

    ; Save our callee and argv pointers, and argc, to scratch registers
    mov   x9, x0
    mov   x11, x1
    mov   x12, x2

    ; Determine if spills are needed
    ; In the common case in which they are not, we perform a tail call
    cmp   x2, #9
    b.hs  L_dyn_call_spill

    ; Calculate the address of the jump table
    adrp x0, L_dyn_call_jt@PAGE
    add x0, x0, L_dyn_call_jt@PAGEOFF
L_dyn_call_no_spill:
    ; We only reach this block if we had no arguments to spill, so
    ; we are not certain about which registers we need to assign. We
    ; simply check for each register whether this a corresponding argument,
    ; and if so, we assign it.

    ; Calculate offset from here to the jump table, then add the number of
    ; arguments we have * 4, then load the destination address from the jump table.
    ; This calculates the block which handles the specific number of registers we
    ; have arguments for, then we jump to that block
    adr     x13, L_dyn_call_no_spill
    ldrsw   x14, [x0, x2, LSL#2]
    add     x13, x13, x14
    br      x13

    ; All of these basic blocks perform a tail call. As such,
    ; the unwinder will skip over this frame should the callee
    ; throw an exception
L_dyn_call_regs0:
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs1:
    ldr x0, [x11]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs2:
    ldp x0, x1, [x11]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs3:
    ldp x0, x1, [x11]
    ldr x2, [x11, #16]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs4:
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs5:
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldr x4, [x11, #32]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs6:
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldp x4, x5, [x11, #32]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs7:
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldp x4, x5, [x11, #32]
    ldr x6, [x11, #48]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_regs8:
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldp x4, x5, [x11, #32]
    ldp x6, x7, [x11, #48]
    ldp x29, x30, [sp, #-16]
    br  x9

L_dyn_call_spill:
    ; If we hit this block, we have identified that there are
    ; arguments to spill. We perform some setup for the actual spilling
    ; We need to start by saving the stack space we stored the original frame pointer/return address in
    sub sp, sp, #16

    ; Calculate spill count for later (rep uses rcx for the iteration count,
    ; which in this case is the number of quadwords to copy)
    mov   x13, #0            ; zero out x13 to use as the loop index
    sub   x14, x12, #8       ; subtract 8 from the argument count, to be the loop bound

    ; Calculate spill space (i.e. new bottom of stack), ensure it is 16-byte aligned, and allocate it
    ;
    ; x12 = 0 + (x14 * 8)
    add  x12, xzr, x14,LSL#3
    ; x12 = sp - x12
    sub  x12, sp, x12
    ; x12 = x12 & ~16
    and  x12, x12, #-16
    mov  sp, x12

L_dyn_call_spill_loop:
    ; if x14 == 0, we're done
    cmp x14, #0
    beq L_dyn_call_spill_loop_end
    ; Calculate source pointer (starting at first spilled element of argv)
    ; x15 = x11 + 8 + (x13 * 8)
    add x15, x11, x13,LSL#3
    add x15, x15, #64
    ; Calculate destination pointer (starting at bottom of spill region)
    ; x16 = sp + (x13 * 8)
    add x16, sp, x13,LSL#3
    ; Copy value from memory given by address in x15 to register x17
    ldr x17, [x15]
    ; Copy value in x17 to memory given by address in x16
    str x17, [x16]
    ;  Decrement the loop bound
    sub x14, x14, #1
    ; Increment the loop index
    add x13, x13, #1
    ; Next iteration
    b L_dyn_call_spill_loop
L_dyn_call_spill_loop_end:
    ; We've spilled arguments, so we have at least 8 args
    ldp x0, x1, [x11]
    ldp x2, x3, [x11, #16]
    ldp x4, x5, [x11, #32]
    ldp x6, x7, [x11, #48]

L_dyn_call_exec:
    ; If we spill arguments to the stack, we can't perform
    ; a tail call, so we do a normal call/ret sequence here
    ;
    ; At this point, the stack should be 16-byte aligned,
    ; all of the callee arguments should be in registers or
    ; spilled to the stack (spilled in reverse order). All
    ; that remains is to actually execute the call!
    ;
    ; This instruction will push the return address and jump,
    ; and we can expect %rbp to be the same as we left it upon
    ; return.
    blr      x9

L_dyn_call_ret:
    ; Non-tail call completed successfully
    ; Our frame pointer should have the same address as our original stack pointer
    ; so we restore it, and then reload the original frame pointer/return address
    mov sp, x29
    ldp x29, x30, [sp, #-16]
    ret

L_dyn_call_end:
    .cfi_endproc

    ; The following is the jump table for setting up calls with
    ; a variable number of register-based arguments
    .p2align 2
L_dyn_call_jt:
    .long L_dyn_call_regs0-L_dyn_call_no_spill
    .long L_dyn_call_regs1-L_dyn_call_no_spill
    .long L_dyn_call_regs2-L_dyn_call_no_spill
    .long L_dyn_call_regs3-L_dyn_call_no_spill
    .long L_dyn_call_regs4-L_dyn_call_no_spill
    .long L_dyn_call_regs5-L_dyn_call_no_spill
    .long L_dyn_call_regs6-L_dyn_call_no_spill
    .long L_dyn_call_regs7-L_dyn_call_no_spill
    .long L_dyn_call_regs8-L_dyn_call_no_spill


    ; The following is the LSDA metadata for exception handling
    .section __TEXT,__gcc_except_tab
    .p2align 2
L_dyn_call_lsda:
    ; Landing pad encoding (DW_EH_PE_omit) = omit
    .byte 255
    ; DWARF encoding of type entries in types table = no type entries
    .byte 255
L_dyn_call_cst_header:
    ; Call site encoding = uleb128
    .byte 1
    ; Size of call site table
    .uleb128 L_dyn_call_cst_end-L_dyn_call_cst_begin
L_dyn_call_cst_begin:
    ; Call site entry for the dynamic callee (offset, size, pad, action)
    ;  call occurs between L_dyn_call_exec and L_dyn_call_ret
    .uleb128 L_dyn_call_exec-L_dyn_call_begin
    .uleb128 L_dyn_call_ret-L_dyn_call_exec
    ;  no landing pad, the unwinder will skip over us
    .byte 0
    ;  offset into action table (0 is no action)
    .byte 0
L_dyn_call_cst_end:
    .p2align 2
