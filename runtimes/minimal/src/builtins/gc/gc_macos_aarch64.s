    .p2align 4
    .global ___lumen_swap_stack
___lumen_builtin_gc.enter:
    .cfi_startproc
    .cfi_personality 155, DW.ref.rust_eh_personality
    ;; We're performing a tail call, but we need to set up the arguments
    ;; First, move the return address into x0
    mov x30, x0
    ;; Then, copy the frame pointer address into x1
    mov x29, x1
    ;; Tail call __lumen_builtin_gc.run, when it returns it will return over us to the caller
    b __lumen_builtin_gc.run

L_builtin_gc_enter_end:
    .cfi_endproc
