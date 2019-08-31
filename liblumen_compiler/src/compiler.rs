use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use failure::{format_err, Error};

use libeir_diagnostics::emitter::{cyan, green, green_bold, white, yellow, yellow_bold};
use libeir_diagnostics::{ColorSpec, Emitter, NullEmitter, StandardStreamEmitter};
use libeir_diagnostics::{Diagnostic, Severity};

use libeir_intern::Ident;
use libeir_ir::Module;

use libeir_passes::PassManager;
use libeir_syntax_erl::{ParseConfig, Parser};

pub use super::config::{CompilerMode, CompilerSettings, Verbosity};
pub use super::errors::CompilerError;

/// The result produced by compiler functions
pub type CompileResult = Result<(), Error>;

pub struct CompilationInfo {
    num_modules: usize,
    compilation_time: usize,
}
impl CompilationInfo {
    pub fn new() -> Self {
        CompilationInfo {
            num_modules: 0,
            compilation_time: 0,
        }
    }
}

pub struct Compiler {
    config: CompilerSettings,
    info: CompilationInfo,
    emitter: Arc<dyn Emitter>,
}
impl Compiler {
    pub fn new(config: CompilerSettings) -> Self {
        let emitter: Arc<dyn Emitter> = match config.verbosity {
            Verbosity::Silent => Arc::new(NullEmitter::new()),
            v => Arc::new(
                StandardStreamEmitter::new(config.color)
                    .set_codemap(config.codemap.clone())
                    .set_min_severity(verbosity_to_severity(v)),
            ),
        };
        let info = CompilationInfo::new();

        Compiler {
            config,
            info,
            emitter,
        }
    }

    pub fn compile(&mut self) -> Result<(), Error> {
        unimplemented!()
    }

    // Parses all modules into a map. The map uses the module name symbol
    // as the key, and the AST for the module as the value.
    fn parse_modules(&mut self) -> Result<HashMap<Ident, Module>, Error> {
        use walkdir::{DirEntry, WalkDir};

        let mut parser = Parser::new(self.config.clone().into());

        let extension = match self.config.mode {
            CompilerMode::Erlang => "erl",
        };

        fn is_hidden(entry: &DirEntry) -> bool {
            entry
                .file_name()
                .to_str()
                .map(|s| s.starts_with("."))
                .unwrap_or(false)
        }

        fn is_source_file(entry: &DirEntry, extension: &str) -> bool {
            if !entry.file_type().is_file() {
                return false;
            }
            match Path::new(entry.file_name()).extension() {
                None => false,
                Some(ext) => ext == extension,
            }
        }

        let walker = WalkDir::new(self.config.source_dir.clone())
            .follow_links(true)
            .into_iter();

        let mut modules = HashMap::new();
        let config = ParseConfig::default();
        let mut parser = Parser::new(config);

        for entry in walker.filter_entry(|e| !is_hidden(e) && is_source_file(e, extension)) {
            let entry = entry.unwrap();
            let file = entry.path();

            let mut module = match self.config.mode {
                CompilerMode::Erlang => self.parse_erl(&mut parser, file)?,
            };

            modules.insert(module.name.clone(), module);
        }

        Ok(modules)
    }

    // Compiles a .erl file to Erlang AST
    fn parse_erl(&self, parser: &mut Parser, file: &Path) -> Result<Module, Error> {
        use libeir_syntax_erl::ast;
        match parser.parse_file::<&Path, ast::Module>(file) {
            Ok(ast) => {
                let (res, messages) = libeir_syntax_erl::lower_module(&ast);
                for msg in messages.iter() {
                    self.diagnostic(&msg.to_diagnostic());
                }
                match res.ok() {
                    Some(mut ir) => {
                        let mut pass_manager = PassManager::default();
                        pass_manager.run(&mut ir);
                        Ok(ir)
                    }
                    None => Err(CompilerError::Failed.into()),
                }
            }
            Err(errs) => Err(CompilerError::Parser {
                codemap: self.config.codemap.clone(),
                errs,
            }
            .into()),
        }
    }

    #[inline]
    fn write_warning<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .warn(Some(color), &message.to_string())
            .unwrap();
    }

    #[inline]
    fn write_info<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .emit(Some(color), &message.to_string())
            .unwrap();
    }

    #[allow(unused)]
    #[inline]
    fn write_debug<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .debug(Some(color), &message.to_string())
            .unwrap();
    }

    pub fn warnings_as_errors(&self) -> bool {
        self.config.warnings_as_errors
    }

    pub fn no_warn(&self) -> bool {
        self.config.no_warn
    }

    pub fn output_dir(&self) -> PathBuf {
        self.config.output_dir.clone()
    }

    pub fn warn<M: Display>(&self, message: M) {
        self.write_warning(yellow_bold(), "WARN: ");
        self.write_warning(yellow(), &message.to_string());
    }

    pub fn info<M: Display>(&self, message: M) {
        self.write_info(cyan(), &message.to_string());
    }

    pub fn debug<M: Display>(&self, message: M) {
        self.write_info(white(), &message.to_string());
    }

    pub fn diagnostic(&self, diagnostic: &Diagnostic) {
        self.emitter.diagnostic(diagnostic).unwrap();
    }
}

fn verbosity_to_severity(v: Verbosity) -> Severity {
    match v {
        Verbosity::Silent => Severity::Bug,
        Verbosity::Error => Severity::Error,
        Verbosity::Warning => Severity::Warning,
        Verbosity::Info => Severity::Note,
        Verbosity::Debug => Severity::Note,
    }
}
