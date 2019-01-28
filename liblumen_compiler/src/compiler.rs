use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use failure::{format_err, Error};

use liblumen_diagnostics::emitter::{cyan, green, green_bold, white, yellow, yellow_bold};
use liblumen_diagnostics::{ColorSpec, Emitter, NullEmitter, StandardStreamEmitter};
use liblumen_diagnostics::{Diagnostic, Severity};

use liblumen_syntax::ast::Module;
use liblumen_syntax::{Parser, Symbol};

use liblumen_codegen as codegen;
use liblumen_common as common;
use liblumen_core as core;

pub use super::config::{CompilerMode, CompilerSettings, Verbosity};
pub use super::errors::CompilerError;
use super::lint;

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
        codegen::initialize();

        let modules = self.parse_modules()?;
        // Perform initial verification of parsed modules
        for (_name, module) in modules.iter() {
            lint::module(self, module)?;
        }
        // Lower from parse tree to Core IR
        let _modules = core::transform(self, modules)?;
        //let modules = semantic_analysis::analyze(&config, core)?;
        //let modules = cps::transform(&config, modules)?;
        //let info = codegen::run(&config, modules, codegen::OutputType::IR)?;

        self.write_info(green_bold(), "Compilation successful!\n");
        self.write_info(
            green(),
            format!(
                "Compiled {} modules in {}",
                self.info.num_modules, self.info.compilation_time
            ),
        );

        Ok(())
    }

    // Parses all modules into a map. The map uses the module name symbol
    // as the key, and the AST for the module as the value.
    fn parse_modules(&mut self) -> Result<HashMap<Symbol, Module>, Error> {
        use walkdir::{DirEntry, WalkDir};

        let mut parser = Parser::new(self.config.clone().into());

        let extension = match self.config.mode {
            CompilerMode::BEAM => "beam",
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

        for entry in walker.filter_entry(|e| !is_hidden(e) && is_source_file(e, extension)) {
            let entry = entry.unwrap();
            let file = entry.path();

            let mut module = match self.config.mode {
                CompilerMode::BEAM => self.parse_beam(file)?,
                CompilerMode::Erlang => self.parse_erl(&mut parser, file)?,
            };

            self.apply_compiler_settings(&mut module);

            modules.insert(module.name.name.clone(), module);
        }

        Ok(modules)
    }

    // Compiles a BEAM file to a Module
    fn parse_beam(&self, _file: &Path) -> Result<Module, Error> {
        Err(format_err!(
            "Currently, compiler support for BEAM files is unimplemented"
        ))
    }

    // Compiles a .erl file to Erlang AST
    fn parse_erl(&self, parser: &mut Parser, file: &Path) -> Result<Module, Error> {
        match parser.parse_file::<&Path, Module>(file) {
            Ok(module) => Ok(module),
            Err(errs) => Err(CompilerError::Parser {
                codemap: self.config.codemap.clone(),
                errs,
            }
            .into()),
        }
    }

    fn apply_compiler_settings(&self, module: &mut Module) {
        module.compile.as_mut().and_then(|mut co| {
            co.warnings_as_errors = self.config.warnings_as_errors;
            co.no_warn = self.config.no_warn;

            Some(co)
        });
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
}
impl common::compiler::Compiler for Compiler {
    fn warnings_as_errors(&self) -> bool {
        self.config.warnings_as_errors
    }

    fn no_warn(&self) -> bool {
        self.config.no_warn
    }

    fn output_dir(&self) -> PathBuf {
        self.config.output_dir.clone()
    }

    fn warn<M: Display>(&self, message: M) {
        self.write_warning(yellow_bold(), "WARN: ");
        self.write_warning(yellow(), &message.to_string());
    }

    fn info<M: Display>(&self, message: M) {
        self.write_info(cyan(), &message.to_string());
    }

    fn debug<M: Display>(&self, message: M) {
        self.write_info(white(), &message.to_string());
    }

    fn diagnostic(&self, diagnostic: &Diagnostic) {
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
