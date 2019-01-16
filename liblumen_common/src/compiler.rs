use std::fmt::Display;
use std::path::PathBuf;

use liblumen_diagnostics::Diagnostic;

/// This trait exposes compiler functions to crates which are dependencies
/// of the compiler, but need to interact with it in various ways.
pub trait Compiler {
    /// Returns whether warnings should be treated as errors
    fn warnings_as_errors(&self) -> bool;

    /// Returns whether warnings are disabled or not
    fn no_warn(&self) -> bool;

    /// Returns the configured output directory
    fn output_dir(&self) -> PathBuf;

    /// Display a warning message to the user
    fn warn<M: Display>(&self, message: M);

    /// Display an informational message to the user
    fn info<M: Display>(&self, message: M);

    /// Display debugging information to the user
    fn debug<M: Display>(&self, message: M);

    /// Display a diagnostic to the user
    fn diagnostic(&self, diagnostic: &Diagnostic);
}
