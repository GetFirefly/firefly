use std::collections::btree_map::{
    Iter as BTreeMapIter, Keys as BTreeMapKeysIter, Values as BTreeMapValuesIter,
};
use std::collections::BTreeMap;
use std::convert::{Into, TryInto};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::ArgMatches;
use thiserror::Error;

use libeir_ir as eir;
use libeir_syntax_erl as syntax;
use liblumen_util::diagnostics::FileName;
use liblumen_util::fs;

use crate::{Input, OptionInfo, Options, ParseOption};

pub trait Emit {
    const TYPE: OutputType;

    fn emit_output_type(&self) -> OutputType {
        Self::TYPE
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()>;
}
impl Emit for syntax::ast::Module {
    const TYPE: OutputType = OutputType::AST;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;
        write!(f, "{:#?}", self)?;
        Ok(())
    }
}
impl Emit for eir::Module {
    const TYPE: OutputType = OutputType::EIR;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;
        let text = self.to_text_standard();
        f.write_all(text.as_bytes())?;
        Ok(())
    }
}

struct OutputTypeSpec {
    output_type: OutputType,
    pattern: Option<String>,
}
impl FromStr for OutputTypeSpec {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split = s.splitn(2, '=').collect::<Vec<_>>();
        let output_type = OutputType::from_str(split[0])?;
        if split.len() == 1 {
            return Ok(Self {
                output_type,
                pattern: None,
            });
        }
        Ok(Self {
            output_type,
            pattern: Some(split[1].to_string()),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum OutputType {
    AST,
    EIR,
    EIRDialect,
    StandardDialect,
    LLVMDialect,
    LLVMAssembly,
    LLVMBitcode,
    Assembly,
    Object,
    Exe,
}
impl FromStr for OutputType {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ast" => Ok(OutputType::AST),
            "eir" => Ok(OutputType::EIR),
            "mlir-eir" => Ok(OutputType::EIRDialect),
            "mlir-std" => Ok(OutputType::StandardDialect),
            "mlir-llvm" => Ok(OutputType::LLVMDialect),
            "llvm-ir" => Ok(OutputType::LLVMAssembly),
            "llvm-bc" => Ok(OutputType::LLVMBitcode),
            "asm" => Ok(OutputType::Assembly),
            "obj" => Ok(OutputType::Object),
            "link" => Ok(OutputType::Exe),
            _ => Err(()),
        }
    }
}
impl AsRef<str> for OutputType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl OutputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            &OutputType::AST => "ast",
            &OutputType::EIR => "eir",
            &OutputType::EIRDialect => "mlir-eir",
            &OutputType::StandardDialect => "mlir-std",
            &OutputType::LLVMDialect => "mlir-llvm",
            &OutputType::LLVMAssembly => "llvm-ir",
            &OutputType::LLVMBitcode => "llvm-bc",
            &OutputType::Assembly => "asm",
            &OutputType::Object => "obj",
            &OutputType::Exe => "link",
        }
    }

    pub fn variants() -> &'static [OutputType] {
        &[
            OutputType::AST,
            OutputType::EIR,
            OutputType::EIRDialect,
            OutputType::StandardDialect,
            OutputType::LLVMDialect,
            OutputType::LLVMAssembly,
            OutputType::LLVMBitcode,
            OutputType::Assembly,
            OutputType::Object,
            OutputType::Exe,
        ]
    }

    pub const fn help() -> &'static str {
        "Comma-separated list of output types for the compiler to generate.\n\
         You may specify one or more types (comma-separated), and each type\n\
         may also include a glob pattern, which filters the inputs for which\n\
         that output type should apply.\n\
         \n\
         Supported output types:\n  \
           all       = Emit everything\n  \
           ast       = Abstract Syntax Tree\n  \
           eir       = Erlang Intermediate Representation\n  \
           mlir-eir  = MLIR (Erlang Dialect)\n  \
           mlir-std  = MLIR (Standard Dialect)\n  \
           mlir-llvm = MLIR (LLVM Dialect)\n  \
           llvm-ir   = LLVM IR\n  \
           llvm-bc   = LLVM Bitcode (*)\n  \
           asm       = Assembly (*)\n  \
           obj       = Object File (*)\n  \
           link      =  Executable (*)\n\
         \n\
         (*) Indicates that globs cannot be applied to this output type"
    }

    pub fn extension(&self) -> &'static str {
        match *self {
            OutputType::AST => "ast",
            OutputType::EIR => "eir",
            OutputType::EIRDialect => "eir.mlir",
            OutputType::StandardDialect => "std.mlir",
            OutputType::LLVMDialect => "llvm.mlir",
            OutputType::LLVMAssembly => "ll",
            OutputType::LLVMBitcode => "bc",
            OutputType::Assembly => "s",
            OutputType::Object => "o",
            OutputType::Exe => "",
        }
    }
}

#[derive(Error, Debug)]
pub enum OutputTypeError {
    #[error("found conflicting output type specifications")]
    Conflict,
    #[error("invalid glob pattern for {output_type} around character {pos}: {message}")]
    InvalidPattern {
        output_type: &'static str,
        pos: usize,
        message: &'static str,
    },
    #[error("invalid output type specification for {output_type}: {message}")]
    Invalid {
        output_type: &'static str,
        message: &'static str,
    },
}
impl From<fs::PatternError> for OutputTypeError {
    fn from(err: fs::PatternError) -> Self {
        Self::InvalidPattern {
            output_type: "<omitted>",
            pos: err.pos,
            message: err.msg,
        }
    }
}
impl Into<clap::Error> for OutputTypeError {
    fn into(self) -> clap::Error {
        clap::Error {
            kind: clap::ErrorKind::InvalidValue,
            message: self.to_string(),
            info: None,
        }
    }
}

/// Use tree-based collections to cheaply get a deterministic `Hash` implementation.
/// *Do not* switch `BTreeMap` out for an unsorted container type! That would break
/// dependency tracking for command-line arguments.
#[derive(Debug, Clone, Hash)]
pub struct OutputTypes(BTreeMap<OutputType, Option<fs::Pattern>>);

impl Default for OutputTypes {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}
impl OutputTypes {
    pub fn new(entries: &[(OutputType, Option<String>)]) -> Result<Self, OutputTypeError> {
        use std::collections::btree_map::Entry;

        let mut map: BTreeMap<OutputType, Option<fs::Pattern>> = BTreeMap::new();
        for (k, ref v) in entries {
            let pattern = match v.as_ref().map(fs::glob) {
                None => None,
                Some(Ok(pattern)) => Some(pattern),
                Some(Err(err)) => {
                    return Err(OutputTypeError::InvalidPattern {
                        output_type: k.as_str(),
                        pos: err.pos,
                        message: err.msg,
                    });
                }
            };
            match map.entry(k.clone()) {
                Entry::Vacant(entry) => match v {
                    &None => {
                        entry.insert(pattern);
                    }
                    &Some(_) => return Err(OutputTypeError::Conflict),
                },
                Entry::Occupied(mut entry) => {
                    let value = entry.get_mut();
                    if value.is_none() {
                        *value = pattern;
                    } else {
                        return Err(OutputTypeError::Conflict);
                    }
                }
            }
        }

        Ok(Self(map))
    }

    pub fn maybe_emit(&self, input: &Input, output_type: OutputType) -> Option<PathBuf> {
        match self.0.get(&output_type) {
            None => None,
            Some(None) => Some(output_filename(input.source_name(), output_type, None)),
            Some(Some(pattern)) => {
                if pattern.matches_path(input.try_into().unwrap()) {
                    Some(output_filename(input.source_name(), output_type, None))
                } else {
                    None
                }
            }
        }
    }

    pub fn always_emit(&self, input: &Input, output_type: OutputType) -> PathBuf {
        output_filename(input.source_name(), output_type, None)
    }

    pub fn get(&self, key: &OutputType) -> Option<&fs::Pattern> {
        self.0.get(key).and_then(|opt| opt.as_ref())
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    pub fn keys(&self) -> BTreeMapKeysIter<'_, OutputType, Option<fs::Pattern>> {
        self.0.keys()
    }

    pub fn values(&self) -> BTreeMapValuesIter<'_, OutputType, Option<fs::Pattern>> {
        self.0.values()
    }

    pub fn iter(&self) -> BTreeMapIter<'_, OutputType, Option<fs::Pattern>> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Returns `true` if any of the output types require codegen or linking.
    pub fn should_generate_mlir(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::AST => false,
            OutputType::EIR => false,
            _ => true,
        })
    }

    pub fn should_generate_llvm(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::AST => false,
            OutputType::EIR => false,
            OutputType::EIRDialect => false,
            OutputType::StandardDialect => false,
            OutputType::LLVMDialect => false,
            _ => true,
        })
    }

    pub fn should_codegen(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::AST => false,
            OutputType::EIR => false,
            OutputType::EIRDialect => false,
            OutputType::StandardDialect => false,
            OutputType::LLVMDialect => false,
            OutputType::LLVMAssembly => false,
            OutputType::LLVMBitcode => false,
            _ => true,
        })
    }
}
impl ParseOption for OutputTypes {
    fn parse_option<'a>(info: &OptionInfo, matches: &ArgMatches<'a>) -> clap::Result<Self> {
        let mut output_types = Vec::new();

        if let Some(values) = matches.values_of(info.name) {
            for value in values {
                if value.starts_with("all") {
                    let split = value.splitn(2, '=').collect::<Vec<_>>();
                    if split.len() == 1 {
                        for v in OutputType::variants() {
                            output_types.push((*v, None));
                        }
                    } else {
                        for v in OutputType::variants() {
                            output_types.push((*v, Some(split[1].to_string())));
                        }
                    }
                    continue;
                }
                match OutputTypeSpec::from_str(value) {
                    Ok(OutputTypeSpec {
                        output_type,
                        pattern,
                    }) => {
                        output_types.push((output_type, pattern));
                    }
                    Err(_) => {
                        return Err(clap::Error {
                            kind: clap::ErrorKind::ValueValidation,
                            message: format!("invalid output type specification, expected format is `TYPE[=PATH]`"),
                            info: Some(vec![info.name.to_string()]),
                        });
                    }
                }
            }
        }

        if output_types.is_empty() {
            output_types.push((OutputType::Object, None));
            output_types.push((OutputType::Exe, None));
        }
        Self::new(output_types.as_slice()).map_err(|err| {
            let mut clap_err: clap::Error = err.into();
            clap_err.info = Some(vec![info.name.to_string()]);
            clap_err
        })
    }
}

pub fn calculate_outputs(
    input: &Input,
    output_dir: &Path,
    options: &Options,
) -> Result<BTreeMap<OutputType, Option<PathBuf>>, OutputTypeError> {
    let mut outputs = BTreeMap::new();
    // Ensure all output types are represented up to and including the final output type
    for variant in OutputType::variants().iter().copied() {
        outputs.insert(variant, None);
    }
    if !options.project_type.is_executable() {
        // The executable output type is not used
        outputs.remove(&OutputType::Exe);
    }

    // For each output type requested, map the given input to that output type,
    // if the output specification applies. For single file outputs (output types
    // which represent the executable, etc.), no output will be mapped, as that
    // is handled elsewhere - all inputs are implicitly part of such outputs.
    //
    // If an output type has no glob, then it applies to all inputs, otherwise the
    // glob filters out inputs which should not produce those outputs. It is not
    // permitted to use globs when reading from stdin, and function will return an
    // error if that is attempted
    for (output_type, glob_opt) in options.output_types.iter() {
        match glob_opt.as_ref() {
            None => {
                // This output type applies to all inputs
                let output = map_input_output(&input, output_type, output_dir);
                outputs.insert(*output_type, output);
            }
            Some(_pattern) if input.is_virtual() => {
                return Err(OutputTypeError::Invalid {
                    output_type: output_type.as_str(),
                    message: "cannot specify output globs when reading from stdin",
                });
            }
            Some(pattern) => {
                if pattern.matches_path(input.try_into().unwrap()) {
                    let output = map_input_output(&input, output_type, output_dir);
                    outputs.insert(*output_type, output);
                }
            }
        }
    }

    Ok(outputs)
}

// Given a specific input, output type, output directory and options; this function produces
// a single output path, if the input should produce an output; otherwise it returns `None`
fn map_input_output(input: &Input, output_type: &OutputType, output_dir: &Path) -> Option<PathBuf> {
    match output_type {
        OutputType::LLVMBitcode | OutputType::Assembly | OutputType::Object | OutputType::Exe => {
            // All inputs go into a single output for these types
            None
        }
        _ => Some(output_filename(
            input.source_name(),
            *output_type,
            Some(output_dir),
        )),
    }
}

fn output_filename(
    source_name: FileName,
    output_type: OutputType,
    output_dir_opt: Option<&Path>,
) -> PathBuf {
    let source_path: &Path = source_name.as_ref();
    let stem = source_path.file_stem().unwrap().to_str().unwrap();
    if let Some(output_dir) = output_dir_opt {
        output_dir
            .join(stem)
            .with_extension(output_type.extension())
    } else {
        PathBuf::from(stem).with_extension(output_type.extension())
    }
}
