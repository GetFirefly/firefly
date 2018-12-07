//! A Rust representation of Abstract Syntax Trees of Erlang modules.
//!
//! Currently, works by loading AST from BEAM files with debug_info enabled
//!
//! # References
//!
//! * [The Abstract Format](http://erlang.org/doc/apps/erts/absform.html)
//!
//! # Examples
//!
//!     use syntax::ast::AST;
//!
//!     let ast = AST::from_beam_file("src/testdata/test.beam").unwrap();
//!     println!("{:?}", ast);
//!
pub mod ast;
pub mod error;
pub mod format;

#[cfg(test)]
mod test;

use std::path::Path;

use self::error::FromBeamError;

pub type FromBeamResult<T> = Result<T, error::FromBeamError>;

/// Abstract Syntax Tree
#[derive(Debug)]
pub struct AST {
    pub module: ast::ModuleDecl,
}
impl AST {
    /// Builds AST from the BEAM file
    pub fn from_beam_file<P: AsRef<Path>>(beam_file: P) -> FromBeamResult<Self> {
        use self::format::raw_abstract_v1::AbstractCode;
        let code = AbstractCode::from_beam_file(beam_file)?;
        let forms = code.to_forms()?;
        Ok(AST {
            module: ast::ModuleDecl { forms: forms },
        })
    }
}
