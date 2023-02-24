///! This module provides a struct that contains metadata about an Erlang application.
///!
///! It implements a limited parser for Erlang application resource files - i.e. `foo.app`
///! or `foo.app.src` - sufficient to provide us with the key details about an Erlang app.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use firefly_intern::Symbol;
use firefly_syntax_pp::ast::{Root, Term};
use firefly_syntax_pp::ParserError;
use firefly_util::diagnostics::*;

type Parser = firefly_parser::Parser<()>;

#[derive(Debug, thiserror::Error)]
pub enum AppResourceError {
    #[error("parsing failed")]
    Parser(#[from] ParserError),

    #[error("invalid application spec")]
    Invalid(SourceSpan),
}
impl ToDiagnostic for AppResourceError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            Self::Parser(err) => err.to_diagnostic(),
            Self::Invalid(span) => Diagnostic::error()
                .with_message("invalid application spec")
                .with_labels(vec![Label::primary(span.source_id(), span)]),
        }
    }
}

/// Metadata about an Erlang application
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct App {
    /// The name of the application
    pub name: Symbol,
    /// The specified version of the application. Not required.
    pub version: Option<String>,
    /// The root directory in which the application was found. Not required.
    pub root: Option<PathBuf>,
    /// The set of modules names contained in this application.
    ///
    /// NOTE: This may not match the full set of modules found on disk, but
    /// is the set of modules which systools would package in a release, so
    /// if they don't match, it is likely a mistake.
    pub modules: Vec<Symbol>,
    /// The full set of applications this application depends on.
    pub applications: Vec<Symbol>,
    /// The set of applications included by this application. These applications
    /// are loaded but not started when this application starts by the application
    /// controller.
    pub included_applications: Vec<Symbol>,
    /// A list of optional `applications`. To ensure optional applcations are started
    /// prior to this application, they must be in both this list and `applications`.
    pub optional_applications: Vec<Symbol>,
    /// Configuration parameters used by the application. The value of a configuration
    /// parameter is retrieved by calling `application:get_env/1,2`. The values in the
    /// application resource file can be overridden by values in a configuration file
    /// or by command-line flags.
    pub env: Vec<(Symbol, Term)>,
    /// For OTP applications (i.e. those with a supervisor tree), this is the
    /// application callback module for this application. If not present, then
    /// this is just a library application
    pub otp_module: Option<Symbol>,
    /// These are the start arguments to be passed to `otp_module:start/2`
    pub start_args: Vec<Term>,
    /// A list of start phases and corresponding start arguments for the application.
    /// If this key is present, the application master, in addition to the usual call
    /// to `otp_module:start/2`, also calls `otp_module:start_phase(Phase, Type, PhaseArgs)`
    /// for each start phase defined in this list. Only after this extended start procedure
    /// does `application:start(Application)` return.
    ///
    /// Start phases can be used to synchronize startup of an application and its included
    /// applications. In this case, `mod` must be specified as follows:
    ///
    /// ```erlang
    ///     {mod, {application_starter,[Module,StartArgs]}}
    /// ```
    ///
    /// The application master then calls `Module:start/2` for the primary application, followed
    /// by calls to `Module:start_phase/3` for each start phase (as defined for the primary application),
    /// both for the primary application and for each of its included applications, for which the start
    /// phase is defined.
    ///
    /// This implies that for an included application, the set of start phases must be a subset of the
    /// set of phases defined for the including application.
    pub start_phases: Vec<(Symbol, Term)>,
    /// A list of application versions that the application depends on, e.g. `"kernel-3.0"`.
    ///
    /// These versions indicate _minimum_ requirements, i.e. a larger version than the one specified
    /// in the dependency satisifies the requirement.
    pub runtime_dependencies: Vec<Symbol>,
}
impl App {
    /// Create a new empty application with the given name
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            version: None,
            root: None,
            modules: vec![],
            applications: vec![],
            included_applications: vec![],
            optional_applications: vec![],
            env: vec![],
            otp_module: None,
            start_args: vec![],
            start_phases: vec![],
            runtime_dependencies: vec![],
        }
    }

    /// Parse an application resource from the given path
    pub fn parse<P: AsRef<Path>>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        path: P,
    ) -> Result<Arc<Self>, AppResourceError> {
        let path = path.as_ref();
        let root = path.parent().unwrap().parent().unwrap().to_path_buf();

        let parser = Parser::new((), codemap.clone());
        match parser.parse_file::<Root, _, ParserError>(diagnostics, path) {
            Ok(parsed) => Self::decode(diagnostics, parsed).map(|mut app| {
                app.root.replace(root);
                Arc::new(app)
            }),
            Err(err) => Err(err.into()),
        }
    }

    /// Parse an application resource from the given string
    ///
    /// NOTE: The resulting manifest will not have `root` set, make sure
    /// you set it manually if the application has a corresponding root
    /// directory
    pub fn parse_str<S: AsRef<str>>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        source: S,
    ) -> Result<Arc<Self>, AppResourceError> {
        let parser = Parser::new((), codemap.clone());
        match parser.parse_string::<Root, _, ParserError>(diagnostics, source) {
            Ok(root) => Self::decode(diagnostics, root).map(Arc::new),
            Err(err) => Err(err.into()),
        }
    }

    fn decode(diagnostics: &DiagnosticsHandler, root: Root) -> Result<App, AppResourceError> {
        // Make sure we have a minimum viable spec
        let mut resource = match root.term {
            Term::Tuple(tuple) => tuple,
            other => {
                let span = other.span();
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected a tuple")
                    .emit();
                return Err(AppResourceError::Invalid(span));
            }
        };
        if resource.len() < 3 {
            let span = resource.span();
            let message = format!("expected a 3-tuple, but found {} elements", resource.len());
            diagnostics
                .diagnostic(Severity::Error)
                .with_message("invalid application spec")
                .with_primary_label(span, message)
                .emit();
            return Err(AppResourceError::Invalid(span));
        }

        let meta = resource.pop().unwrap();
        let name = resource.pop().unwrap();

        // We expect the tuple to be tagged 'applciation'
        {
            let tag = resource.pop().unwrap();
            let span = tag.span();
            if !tag
                .as_atom()
                .map(|a| a.item == "application")
                .unwrap_or_default()
            {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected atom 'application'")
                    .emit();
                return Err(AppResourceError::Invalid(span));
            }
        }

        // We expect an atom as the second element in the tuple
        let name = {
            let span = name.span();
            name.as_atom().map_err(|_| {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected atom")
                    .emit();
                AppResourceError::Invalid(span)
            })?
        };

        // Initialize default app metadata with the parsed name
        let mut app = App::new(name.item);

        // We expect the third element to be a (possibly empty) list
        let mut meta = {
            let span = meta.span();
            meta.as_list().map_err(|_| {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected list")
                    .emit();
                AppResourceError::Invalid(span)
            })?
        };

        // Iterate over the application metadata, handling keys we are interested in
        for item in meta.item.drain(..) {
            let span = item.span();
            let mut item = item.as_tuple().map_err(|_| {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected tuple")
                    .emit();
                AppResourceError::Invalid(span)
            })?;
            // Must be a valid keyword item
            if item.len() != 2 {
                let span = item.span();
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid application spec")
                    .with_primary_label(span, "expected keyword list item (i.e. 2-tuple)")
                    .emit();
                return Err(AppResourceError::Invalid(span));
            }
            // Keys must be atoms
            let value = item.pop().unwrap();
            let key = {
                let key = item.pop().unwrap();
                let span = key.span();
                key.as_atom().map_err(|_| {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid application spec")
                        .with_primary_label(span, "expected atom here")
                        .emit();
                    AppResourceError::Invalid(span)
                })?
            };
            match key.as_str().get() {
                "vsn" => {
                    app.version.replace(value.as_string().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected string")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?);
                }
                "modules" => {
                    let mut modules = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for module in modules.drain(..) {
                        app.modules
                            .push(module.as_atom().map(|a| a.item).map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected module name as an atom")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?);
                    }
                }
                "applications" => {
                    let mut applications = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for application in applications.drain(..) {
                        app.applications
                            .push(application.as_atom().map(|a| a.item).map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(
                                        span,
                                        "expected application name as an atom",
                                    )
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?);
                    }
                }
                "included_applications" => {
                    let mut applications = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for application in applications.drain(..) {
                        app.included_applications.push(
                            application.as_atom().map(|a| a.item).map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(
                                        span,
                                        "expected application name as an atom",
                                    )
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?,
                        );
                    }
                }
                "optional_applications" => {
                    let mut applications = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for application in applications.drain(..) {
                        app.optional_applications.push(
                            application.as_atom().map(|a| a.item).map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(
                                        span,
                                        "expected application name as an atom",
                                    )
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?,
                        );
                    }
                }
                "runtime_dependencies" => {
                    let mut runtime_dependencies = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for runtime_dependency in runtime_dependencies.drain(..) {
                        app.runtime_dependencies
                            .push(runtime_dependency.as_string_symbol().map(|a| a.item).map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics.diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected application version string, e.g. \"kernel-3.0\"")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?);
                    }
                }
                "mod" => {
                    let mut mod_tuple = value.as_tuple().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected tuple")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    if mod_tuple.len() != 2 {
                        let span = mod_tuple.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(
                                span,
                                "'mod' key must be a tuple of `{module(), any()}`",
                            )
                            .emit();
                        return Err(AppResourceError::Invalid(span));
                    }
                    let start_arg = mod_tuple.pop().unwrap();
                    let module_name =
                        mod_tuple
                            .pop()
                            .unwrap()
                            .as_atom()
                            .map(|a| a.item)
                            .map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected module name as an atom")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?;
                    app.otp_module.replace(module_name);
                    app.start_args.push(start_arg);
                }
                "env" => {
                    let mut env = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for env_tuple in env.drain(..) {
                        let mut env_tuple = env_tuple.as_tuple().map_err(|invalid| {
                            let span = invalid.span();
                            diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("invalid application spec")
                                .with_primary_label(span, "expected tuple of {atom(), term()}")
                                .emit();
                            AppResourceError::Invalid(span)
                        })?;
                        if env_tuple.len() != 2 {
                            let span = env_tuple.span();
                            diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("invalid application spec")
                                .with_primary_label(span, "expected tuple of {atom(), term()}")
                                .emit();
                            return Err(AppResourceError::Invalid(span));
                        }
                        let value = env_tuple.item.pop().unwrap();
                        let key = env_tuple
                            .item
                            .pop()
                            .unwrap()
                            .as_atom()
                            .map(|a| a.item)
                            .map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected env key as an atom")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?;
                        app.env.push((key, value));
                    }
                }
                "start_phases" => {
                    let mut start_phases = value.as_list().map_err(|invalid| {
                        let span = invalid.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid application spec")
                            .with_primary_label(span, "expected list")
                            .emit();
                        AppResourceError::Invalid(span)
                    })?;
                    for start_phase_tuple in start_phases.drain(..) {
                        let mut start_phase_tuple =
                            start_phase_tuple.as_tuple().map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected tuple of {atom(), term()}")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?;
                        if start_phase_tuple.len() != 2 {
                            let span = start_phase_tuple.span();
                            diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("invalid application spec")
                                .with_primary_label(span, "expected tuple of {atom(), term()}")
                                .emit();
                            return Err(AppResourceError::Invalid(span));
                        }
                        let value = start_phase_tuple.item.pop().unwrap();
                        let key = start_phase_tuple
                            .item
                            .pop()
                            .unwrap()
                            .as_atom()
                            .map(|a| a.item)
                            .map_err(|invalid| {
                                let span = invalid.span();
                                diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("invalid application spec")
                                    .with_primary_label(span, "expected start phase as an atom")
                                    .emit();
                                AppResourceError::Invalid(span)
                            })?;
                        app.start_phases.push((key, value));
                    }
                }
                _ => continue,
            }
        }

        Ok(app)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const SIMPLE: &'static str = "{application, simple, []}.";
    const RICH: &'static str = r#"
%% A multi-line
%% comment
{application, example,
  [{description, "An example application"},
   {vsn, "0.1.0-rc0"},
   %% Another comment
   {modules, [example_app, example_sup, example_worker]},
   {registered, [example_registry]},
   {applications, [kernel, stdlib, sasl]},
   {mod, {example_app, []}}
  ]}.
"#;
    const NOT_EVEN_A_RESOURCE: &'static str = r#""hi"."#;
    const INVALID_APP_RESOURCE: &'static str = "{}.";
    const MISSING_TRAILING_DOT: &'static str = "{application, simple, []}";

    fn parse(source: &str) -> Arc<App> {
        let emitter = Arc::new(DefaultEmitter::new(ColorChoice::Auto));
        let codemap = Arc::new(CodeMap::new());
        let diagnostics =
            DiagnosticsHandler::new(DiagnosticsConfig::default(), codemap.clone(), emitter);
        match App::parse_str(&diagnostics, codemap.clone(), source) {
            Ok(parsed) => parsed,
            Err(err) => {
                diagnostics.error(err);
                panic!("parsing failed")
            }
        }
    }

    #[test]
    fn simple_app_resource_test() {
        let app = parse(SIMPLE);
        let name = app.name.as_str().get();
        assert_eq!(name, "simple");
    }

    #[test]
    fn rich_app_resource_test() {
        let app = parse(RICH);
        let name = app.name.as_str().get();
        assert_eq!(name, "example");
        assert_eq!(app.version.as_ref().map(|s| s.as_str()), Some("0.1.0-rc0"));
        assert_eq!(app.modules.len(), 3);
        assert_eq!(app.applications.len(), 3);
        assert_eq!(
            app.otp_module.map(|s| s.as_str().get()),
            Some("example_app")
        );
    }

    #[test]
    #[should_panic(expected = "expected a tuple")]
    fn invalid_manifest_not_even_a_resource() {
        parse(NOT_EVEN_A_RESOURCE);
    }

    #[test]
    #[should_panic(expected = "expected a 3-tuple, but found 0 elements")]
    fn invalid_manifest_invalid_resource() {
        parse(INVALID_APP_RESOURCE);
    }

    #[test]
    #[should_panic(expected = "unexpected end of file")]
    fn invalid_manifest_missing_trailing_dot() {
        parse(MISSING_TRAILING_DOT);
    }
}
