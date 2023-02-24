use std::path::Path;
use std::sync::Arc;

use anyhow::bail;

use firefly_intern::symbols;
use firefly_pass::Pass;
use firefly_session::{Input, InputType, Options};
use firefly_syntax_base::{Deprecation, ModuleMetadata};
use firefly_syntax_erl::{self as syntax_erl, ParseConfig};
use firefly_util::diagnostics::*;

use crate::compiler::Artifact;

#[derive(Clone)]
pub struct ParsePipeline {
    options: Arc<Options>,
    codemap: Arc<CodeMap>,
    diagnostics: Arc<DiagnosticsHandler>,
    config: ParseConfig,
}
impl ParsePipeline {
    pub fn new(
        options: Arc<Options>,
        codemap: Arc<CodeMap>,
        diagnostics: Arc<DiagnosticsHandler>,
    ) -> Self {
        let mut config = ParseConfig::new();
        config.warnings_as_errors = options.warnings_as_errors;
        config.no_warn = options.no_warn;
        config.include_paths = options.include_path.clone();
        config.code_paths = Default::default();
        config.define(symbols::VSN, crate::FIREFLY_RELEASE);
        config.define(symbols::COMPILER_VSN, crate::FIREFLY_RELEASE);

        Self {
            options,
            codemap,
            diagnostics,
            config,
        }
    }
}
impl Pass for ParsePipeline {
    type Input<'a> = Input;
    type Output<'a> = Artifact<syntax_erl::Module, ModuleMetadata>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut passes = ParseInput::new(
            &self.options,
            &self.diagnostics,
            &self.config,
            self.codemap.clone(),
        )
        .chain(AnalyzeMetadata::new(&self.diagnostics));

        passes.run(input)
    }
}

struct ParseInput<'p> {
    options: &'p Options,
    codemap: Arc<CodeMap>,
    diagnostics: &'p DiagnosticsHandler,
    config: &'p ParseConfig,
}
impl<'p> ParseInput<'p> {
    fn new(
        options: &'p Options,
        diagnostics: &'p DiagnosticsHandler,
        config: &'p ParseConfig,
        codemap: Arc<CodeMap>,
    ) -> Self {
        Self {
            options,
            codemap,
            diagnostics,
            config,
        }
    }
}
impl<'p> Pass for ParseInput<'p> {
    type Input<'a> = Input;
    type Output<'a> = Artifact<syntax_erl::Module>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        use firefly_beam::AbstractCode;
        use firefly_parser as parse;
        use firefly_syntax_erl::passes::AbstractErlangToAst;
        use firefly_syntax_pp as syntax_pp;

        match input.get_type() {
            // For standard Erlang sources, we need only parse the source file
            InputType::Erlang => {
                let parser = parse::Parser::new(self.config.clone(), self.codemap.clone());
                let result = match input {
                    Input::File(ref path) => {
                        parser.parse_file::<syntax_erl::Module, &Path, _>(self.diagnostics, path)
                    }
                    Input::Str { ref input, .. } => {
                        parser.parse_string::<syntax_erl::Module, _, _>(self.diagnostics, input)
                    }
                };

                match result {
                    Ok(module) => {
                        let artifact = Artifact {
                            input,
                            output: module,
                            metadata: (),
                        };
                        artifact.maybe_emit_file_with_opts(self.options)?;
                        Ok(artifact)
                    }
                    Err(e) => {
                        self.diagnostics.emit(e);
                        bail!("parsing failed, see diagnostics for details");
                    }
                }
            }
            // For Abstract Erlang, either in source form or from a BEAM, we need to obtain the
            // Abstract Erlang syntax tree, and then convert it to our normal Erlang syntax tree
            InputType::AbstractErlang => {
                let parser = parse::Parser::new((), self.codemap.clone());
                let module = match input {
                    Input::File(ref path) => parser
                        .parse_file::<syntax_pp::ast::Ast, &Path, _>(&self.diagnostics, path)?,
                    Input::Str { ref input, .. } => parser
                        .parse_string::<syntax_pp::ast::Ast, _, _>(&self.diagnostics, input)?,
                };
                let mut passes = AbstractErlangToAst::new(&self.diagnostics, &self.codemap);
                let output = passes.run(module)?;
                let artifact = Artifact {
                    input,
                    output,
                    metadata: (),
                };
                artifact.maybe_emit_file_with_opts(self.options)?;
                Ok(artifact)
            }
            InputType::BEAM => match input {
                Input::File(ref path) => {
                    let module = AbstractCode::from_beam_file(path).map(|code| code.into())?;
                    let mut passes = AbstractErlangToAst::new(&self.diagnostics, &self.codemap);
                    let output = passes.run(module)?;
                    let artifact = Artifact {
                        input,
                        output,
                        metadata: (),
                    };
                    artifact.maybe_emit_file_with_opts(self.options)?;
                    Ok(artifact)
                }
                Input::Str { .. } => {
                    bail!("beam parsing is only supported on files");
                }
            },
            ty => bail!("invalid input type: {}", ty),
        }
    }
}

struct AnalyzeMetadata<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
impl<'p> AnalyzeMetadata<'p> {
    fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
impl<'p> Pass for AnalyzeMetadata<'p> {
    type Input<'a> = Artifact<syntax_erl::Module>;
    type Output<'a> = Artifact<syntax_erl::Module, ModuleMetadata>;

    fn run<'a>(&mut self, artifact: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let Artifact {
            input,
            output: module,
            ..
        } = artifact;

        let name = module.name;
        let mut metadata = ModuleMetadata::new(module.name);
        metadata.exports = module.exports.iter().cloned().collect();
        metadata.deprecation = module.deprecation.clone();
        for dep in module.deprecations.iter().copied() {
            match dep {
                d @ Deprecation::Module { .. } if metadata.deprecation.is_none() => {
                    metadata.deprecation = Some(d);
                    continue;
                }
                Deprecation::Module { .. } => continue,
                Deprecation::FunctionAnyArity {
                    span,
                    name: deprecated_name,
                    flag,
                } => {
                    // Search for matching functions and deprecate them
                    for function in module.functions.keys().copied() {
                        if function.function == deprecated_name {
                            metadata.deprecations.insert(
                                function.resolve(name.name),
                                Deprecation::Function {
                                    span,
                                    function: Span::new(span, function),
                                    flag,
                                },
                            );
                        }
                    }
                }
                Deprecation::Function {
                    span,
                    function,
                    flag,
                } => {
                    if function.is_local() {
                        metadata.deprecations.insert(
                            function.resolve(name.name),
                            Deprecation::Function {
                                span,
                                function,
                                flag,
                            },
                        );
                    } else {
                        let module = function.module.unwrap();
                        if module == name.name {
                            metadata.deprecations.insert(
                                *function,
                                Deprecation::Function {
                                    span,
                                    function,
                                    flag,
                                },
                            );
                        } else {
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("invalid deprecation")
                                .with_primary_label(
                                    span,
                                    "cannot deprecate a function in another module",
                                )
                                .emit();
                        }
                    }
                }
            }
        }

        Ok(Artifact {
            input,
            output: module,
            metadata,
        })
    }
}
