extern "C" {
    pub fn LLVMLumenPrintPasses();
}

pub fn print_passes() {
    // Can be called without initializing LLVM
    unsafe {
        LLVMLumenPrintPasses();
    }
}
