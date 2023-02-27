    .p2align 4
    .global ___firefly_dynamic_apply
___firefly_dynamic_apply:
L_dyn_call_begin:
    .cfi_startproc
    .cfi_personality 155, _rust_eh_personality
    .cfi_lsda 16, L_dyn_call_lsda
    # At this point, the following registers are bound:
    #
    #   rdi <- callee
    #   rsi <- process
    #   rdx <- argv
    #   rcx <- argc
    #
    # Save the parent base pointer for when control returns to this call frame.
    # CFA directives will inform the unwinder to expect rbp at the bottom of the
    # stack for this frame, so this should be the last value on the stack in the caller
    push rbp
    .cfi_def_cfa_offset 16
    .cfi_offset rbp, -16
    mov  rbp, rsp
    .cfi_def_cfa_register rbp

    # Pin callee pointer to r10
    mov  r10, rdi
    # Pin the argv pointer to r11
    mov  r11, rdx
    # The process pointer needs to be in rdi
    mov  rdi, rsi

    # Determine if spills are needed (argc + 1 should be <= 8 when not needed)
    # In the common case in which they are not, we perform a tail call
    cmp  rcx, 6
    ja L_dyn_call_spill

L_dyn_call_no_spill:
    # We only reach this block if we had no arguments to spill, so
    # we are not certain about which registers we need to assign. We
    # simply check for each register whether this a corresponding argument,
    # and if so, we assign it.

    # Calculate offset in jump table to block which handles the specific
    # number of registers we have arguments for, then jump to that block
    mov  rax, rcx
    lea  rcx, [rip + L_dyn_call_jt]
    movsxd  rax, dword ptr [rcx + 4*rax]
    add  rax, rcx
    jmp  rax

    # All of these basic blocks perform a tail call. As such,
    # the unwinder will skip over this frame should the callee
    # throw an exception
L_dyn_call_regs0:
    pop  rbp
    jmp  r10

L_dyn_call_regs1:
    mov  rsi, [r11]
    pop  rbp
    jmp  r10

L_dyn_call_regs2:
    mov  rsi, [r11]
    mov  rdx, [r11 + 8]
    pop  rbp
    jmp  r10

L_dyn_call_regs3:
    mov  rsi, [r11]
    mov  rdx, [r11 + 8]
    mov  rcx, [r11 + 16]
    pop  rbp
    jmp  r10

L_dyn_call_regs4:
    mov  rsi, [r11]
    mov  rdx, [r11 + 8]
    mov  rcx, [r11 + 16]
    mov  r8, [r11 + 24]
    pop  rbp
    jmp  r10

L_dyn_call_regs5:
    mov  rsi, [r11]
    mov  rdx, [r11 + 8]
    mov  rcx, [r11 + 16]
    mov  r8, [r11 + 24]
    mov  r9, [r11 + 32]
    pop  rbp
    jmp  r10

L_dyn_call_spill:
    # If we hit this block, we have identified that there are
    # arguments to spill. We perform some setup for the actual
    # spilling, which is a loop built on `rep movsq`
    #
    # At this point, the following registers are occupied/hold these values:
    #
    #  r10 <- callee
    #  rdi <- process
    #  r11 <- argv
    #  rcx <- argc

    # rcx, rdi, and rsi are used by `rep movsq`, so save them temporarily
    mov  r8, rcx
    mov  r9, rdi

    # Calculate spill count for later (rep uses rcx for the iteration count `i`,
    # which in this case is the number of quadwords to copy)
    sub  rcx, 6

    # Calculate spill space, and ensure it is rounded up to the nearest 16 bytes.
    lea  rax, [rcx * 8 + 15]
    and  rax, -16

    # Allocate spill space
    sub  rsp, rax

    # load source pointer (last item of argv)
    lea  rsi, [r11 + r8 * 8 - 8]
    # load destination pointer (top of spill region)
    lea  rdi, [rsp + rcx * 8 - 8]
    # copy `i` quadwords from source to destination, in reverse
    std
    rep  movsq
    cld

    # We've spilled arguments, so we have at least 6 args, move them into their
    # final destination registers in preparation for the call
    mov  rdi, r9
    mov  rsi, [r11]
    mov  rdx, [r11 + 8]
    mov  rcx, [r11 + 16]
    mov  r8,  [r11 + 24]
    mov  r9,  [r11 + 32]

L_dyn_call_exec:
    # If we spill arguments to the stack, we can't perform
    # a tail call, so we do a normal call/ret sequence here
    #
    # At this point, the stack should be 16-byte aligned,
    # all of the callee arguments should be in registers or
    # spilled to the stack (spilled in reverse order). All
    # that remains is to actually execute the call!
    #
    # This instruction will push the return address and jump,
    # and we can expect rbp to be the same as we left it upon
    # return.
    call  r10

L_dyn_call_ret:
    # Non-tail call completed successfully
    mov  rsp, rbp
    pop  rbp
    ret

L_dyn_call_end:
    .cfi_endproc

    # The following is the jump table for setting up calls with
    # a variable number of register-based arguments
    .p2align 2
    .data_region jt32
    .set L_dyn_call_jt_entry0, L_dyn_call_regs0-L_dyn_call_jt
    .set L_dyn_call_jt_entry1, L_dyn_call_regs1-L_dyn_call_jt
    .set L_dyn_call_jt_entry2, L_dyn_call_regs2-L_dyn_call_jt
    .set L_dyn_call_jt_entry3, L_dyn_call_regs3-L_dyn_call_jt
    .set L_dyn_call_jt_entry4, L_dyn_call_regs4-L_dyn_call_jt
    .set L_dyn_call_jt_entry5, L_dyn_call_regs5-L_dyn_call_jt
L_dyn_call_jt:
    .long L_dyn_call_jt_entry0
    .long L_dyn_call_jt_entry1
    .long L_dyn_call_jt_entry2
    .long L_dyn_call_jt_entry3
    .long L_dyn_call_jt_entry4
    .long L_dyn_call_jt_entry5
    .end_data_region

    # The following is the LSDA metadata for exception handling
    .section __TEXT,__gcc_except_tab
    .p2align 2
L_dyn_call_lsda:
    # Landing pad encoding (DW_EH_PE_omit) = omit
    .byte 255
    # DWARF encoding of type entries in types table = no type entries
    .byte 255
L_dyn_call_cst_header:
    # Call site encoding = uleb128
    .byte 1
    # Size of call site table
    .uleb128 L_dyn_call_cst_end-L_dyn_call_cst_begin
L_dyn_call_cst_begin:
    # Call site entry for the dynamic callee (offset, size, pad, action)
    #  call occurs between L_dyn_call_exec and L_dyn_call_ret
    .uleb128 L_dyn_call_exec-L_dyn_call_begin
    .uleb128 L_dyn_call_ret-L_dyn_call_exec
    #  no landing pad, the unwinder will skip over us
    .byte 0
    #  offset into action table (0 is no action)
    .byte 0
L_dyn_call_cst_end:
    .p2align 2
