/// A type is emittable when there is a canonical textual/binary representation 
/// which can be written to a file with a well-known file type.
///
/// Examples of this relevant to Firefly are:
///
/// * MLIR IR, which has a canonical textual format, typically with the `.mlir` extension
///   * NOTE: MLIR IR consists of one or more dialects, each of which may use its own extension to
///   distinguish itself, but they are ultimately all represented using the canonical form.
/// * LLVM Assembly, which has a canonical textual format with the `.ll` extension
/// * LLVM Bitcode, which has a canonical binary format with the `.bc` extension
/// * Native Assembly, which has a canonical textual format with the `.asm` extension
/// * Object Files, which have a canonical binary format with the `.o` extension
/// * Static Libraries, which have a canonical binary format with the `.a` extension
/// * Dynamic Libraries, which have a canonical binary format with a platform-specific extension
/// * Executables, which have a canonical binary format with platform-specific extension
///
pub trait Emit {
    /// Returns the extension to which this object is associated
    ///
    /// If no extension is known, or an extension is not desired, you may return None.
    fn file_type(&self) -> Option<&'static str> {
        None
    }

    /// Emits the content represented by this object to the given file
    ///
    /// An implementation should try to ensure that the content produced matches the
    /// file type returned above.
    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()>;
}
