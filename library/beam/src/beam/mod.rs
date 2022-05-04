//! BEAM is the file format used in `.beam` files by the BEAM VM.
//!
//! The BEAM format is a binary chunked file format containing multiple sections.  It starts with
//! the magic file constant `FOR1`, which is based on the `FORM` header used in the original
//! [Interchange File Format (IFF)](https://en.wikipedia.org/wiki/Interchange_File_Format) that the
//! BEAM file format is based.
//!
//! After the magic file constant, the format follows normal
//! [Type-Length-Value (TLV)](https://en.wikipedia.org/wiki/Type-length-value) rules, and so
//! it lists the size of rest of the file as a `u32`.
//!
//! After the the length, the first chunk is `BEAM`, which contains all other chunks.
//!
//! ## References
//!
//! * [BEAM Wisdom - BEAM File Format](http://beam-wisdoms.clau.se/en/latest/indepth-beam-file.html#)
//! * [The BEAM Book - The BEAM File Format](https://happi.github.io/theBeamBook/#BEAM_files)
//!
//! ## Alternative Implementations
//!
//! * [org.elixir_lang.beam.Beam in IntelliJ Elixir](https://github.
//!   com/KronicDeth/intellij-elixir/blob/master/src/org/elixir_lang/beam/Beam.kt) in Kotlin
mod code;
mod errors;
mod reader;

pub use self::code::AbstractCode;
pub use self::errors::*;
pub use self::reader::*;
