#[derive(Context)]
pub enum CodeGenContext {}
pub type CodeGenContextRef = *mut CodeGenContext;

extern "C" {
    pub fn CodeGenContextCreate(argc: libc::c_uint, argv: *const *const libc::c_char) -> CodeGenContextRef;
    pub fn CodeGenContextDispose(C: CodeGenContextRef);
}