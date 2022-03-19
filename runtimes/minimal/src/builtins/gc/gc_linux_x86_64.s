    .section .text.__lumen_builtin_gc.enter,"ax",@progbits
    .globl __lumen_builtin_gc.enter
    .p2align 4
    .type __lumen_builtin_gc.enter,@function
__lumen_builtin_gc.enter:
    .cfi_startproc
    .cfi_personality 155, DW.ref.rust_eh_personality
    ;; We're performing a tail call, but we need to set up the arguments
    ;; First, move the return address into %rdi
    popq %rdi
    ;; Then, copy the frame pointer address into %rsi
    movq %rbp, %rsi
    ;; Tail call __lumen_builtin_gc.run, when it returns it will return over us to the caller
    pushq %rdi
    jmp __lumen_builtin_gc.run

.L_builtin_gc_enter_end:
    .size __lumen_builtin_gc.enter, .L_builtin_gc_enter_end-__lumen_builtin_gc.enter
    .cfi_endproc
