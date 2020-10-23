    .p2align 4
    .global ___lumen_dynamic_apply
___lumen_dynamic_apply:
L_dyn_call_begin:
    .cfi_startproc
    .cfi_personality 155, _rust_eh_personality
    .cfi_lsda 16, L_dyn_call_lsda
    # At this point, the following registers are bound:
    #
    #   %rdi <- callee
    #   %rsi <- argv
    #   %rdx <- argc
    #
    # Save the parent base pointer for when control returns to this call frame.
    # CFA directives will inform the unwinder to expect %rbp at the bottom of the
    # stack for this frame, so this should be the last value on the stack in the caller
    pushq %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq  %rsp, %rbp
    .cfi_def_cfa_register %rbp

    # Save our callee and argv pointers, and argc
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax

    # Determine if spills are needed
    # In the common case in which they are not, we perform a tail call
    cmpq  $7, %rdx
    ja L_dyn_call_spill
    
L_dyn_call_no_spill:
    # We only reach this block if we had no arguments to spill, so
    # we are not certain about which registers we need to assign. We
    # simply check for each register whether this a corresponding argument,
    # and if so, we assign it.
    #
    # Sure would be nice if we had the equivalent of LDM from ARM

    # Calculate offset in jump table to block which handles the specific
    # number of registers we have arguments for, then jump to that block
    leaq    L_dyn_call_jt(%rip), %rcx
    movslq  (%rcx,%rax,4), %rax
    addq    %rcx, %rax
    jmpq    *%rax

    # All of these basic blocks perform a tail call. As such,
    # the unwinder will skip over this frame should the callee
    # throw an exception
L_dyn_call_regs0:
    popq %rbp
    jmpq *%r10

L_dyn_call_regs1:
    movq (%r11), %rdi
    popq %rbp
    jmpq *%r10

L_dyn_call_regs2:
    movq (%r11), %rdi
    movq 8(%r11), %rsi
    popq %rbp
    jmpq *%r10

L_dyn_call_regs3:
    movq (%r11), %rdi
    movq 8(%r11), %rsi
    movq 16(%r11), %rdx
    popq %rbp
    jmpq *%r10

L_dyn_call_regs4:
    movq (%r11), %rdi
    movq 8(%r11), %rsi
    movq 16(%r11), %rdx
    movq 24(%r11), %rcx
    popq %rbp
    jmpq *%r10

L_dyn_call_regs5:
    movq (%r11), %rdi
    movq 8(%r11), %rsi
    movq 16(%r11), %rdx
    movq 24(%r11), %rcx
    movq 32(%r11), %r8
    popq %rbp
    jmpq *%r10

L_dyn_call_regs6:
    movq (%r11), %rdi
    movq 8(%r11), %rsi
    movq 16(%r11), %rdx
    movq 24(%r11), %rcx
    movq 32(%r11), %r8
    movq 40(%r11), %r9
    popq %rbp
    jmpq *%r10

L_dyn_call_spill:
    # If we hit this block, we have identified that there are
    # arguments to spill. We perform some setup for the actual
    # spilling, which is a loop built on `rep movsq`

    # Calculate spill count for later (rep uses rcx for the iteration count,
    # which in this case is the number of quadwords to copy)
    movq  %rdx, %rcx
    subq  $6, %rcx

    # Calculate spill space, and ensure it is rounded up to the nearest 16 bytes.
    leaq 15(,%rcx,8), %rax
    andq $-16, %rax

    # Allocate spill space
    subq %rax, %rsp

    # load source pointer (last item of argv)
    leaq -8(%r11, %rdx, 8), %rsi
    # load destination pointer (top of spill region)
    leaq -8(%rsp, %rcx, 8), %rdi
    # copy rcx quadwords from rsi to rdi, in reverse
    std
    rep movsq
    cld

    # We've spilled arguments, so we have at least 6 args
    movq  (%r11),   %rdi
    movq  8(%r11),  %rsi
    movq  16(%r11), %rdx
    movq  24(%r11), %rcx
    movq  32(%r11), %r8
    movq  40(%r11), %r9

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
    # and we can expect %rbp to be the same as we left it upon
    # return.
    callq    *%r10
    
L_dyn_call_ret:
    # Non-tail call completed successfully
    movq %rbp, %rsp
    popq %rbp
    retq

L_dyn_call_end:
    .cfi_endproc

    # The following is the jump table for setting up calls with
    # a variable number of register-based arguments
    .p2align 2
    .data_region jt32
    .set L_dyn_call_jt_entry0, L_dyn_call_exec-L_dyn_call_jt
    .set L_dyn_call_jt_entry1, L_dyn_call_regs1-L_dyn_call_jt
    .set L_dyn_call_jt_entry2, L_dyn_call_regs2-L_dyn_call_jt
    .set L_dyn_call_jt_entry3, L_dyn_call_regs3-L_dyn_call_jt
    .set L_dyn_call_jt_entry4, L_dyn_call_regs4-L_dyn_call_jt
    .set L_dyn_call_jt_entry5, L_dyn_call_regs5-L_dyn_call_jt
    .set L_dyn_call_jt_entry6, L_dyn_call_regs6-L_dyn_call_jt
L_dyn_call_jt:
    .long L_dyn_call_jt_entry0
    .long L_dyn_call_jt_entry1
    .long L_dyn_call_jt_entry2
    .long L_dyn_call_jt_entry3
    .long L_dyn_call_jt_entry4
    .long L_dyn_call_jt_entry5
    .long L_dyn_call_jt_entry6
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
