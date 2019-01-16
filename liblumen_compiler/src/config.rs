use std::str::FromStr;
use std::convert::Into;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use failure::{format_err, Error};

use liblumen_diagnostics::{CodeMap, ColorChoice};
use liblumen_syntax::{Symbol, MacroDef, ParseConfig};

/// Determines which type of compilation to perform,
/// either parsing modules from BEAM files, or by
/// parsing modules from Erlang source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum CompilerMode {
    BEAM,
    Erlang
}
impl FromStr for CompilerMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "beam" => Ok(CompilerMode::BEAM),
            "erl" => Ok(CompilerMode::Erlang),
            _ => Err(format_err!("invalid file type {}", s))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Debug,
    Info,
    Warning,
    Error,
    Silent,
}
impl Verbosity {
    pub fn from_level(level: isize) -> Self {
        if level < 0 {
            return Verbosity::Silent;
        }

        match level {
            0 => Verbosity::Warning,
            1 => Verbosity::Info,
            _ => Verbosity::Debug,
        }
    }
}

/// This structure holds all top-level compiler options
/// and configuration; it is passed through all phases
/// of compilation, including parsing.
#[derive(Debug, Clone)]
pub struct CompilerSettings {
    pub mode: CompilerMode,
    pub color: ColorChoice,
    pub source_dir: PathBuf,
    pub output_dir: PathBuf,
    pub defines: HashMap<Symbol, MacroDef>,
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub verbosity: Verbosity,
    pub code_path: Vec<PathBuf>,
    pub codemap: Arc<Mutex<CodeMap>>,
}
impl Into<ParseConfig> for CompilerSettings {
    fn into(self) -> ParseConfig {
        ParseConfig {
            codemap: self.codemap.clone(),
            warnings_as_errors: self.warnings_as_errors,
            no_warn: self.no_warn,
            code_paths: self.code_path.clone().into(),
            macros: Some(self.defines.clone()),
        }
    }
}
