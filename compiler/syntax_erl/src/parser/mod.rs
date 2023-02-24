/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => {
        SourceSpan::new($l, $r)
    };
    ($i:expr) => {
        SourceSpan::new($i, $i)
    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
#[allow(unused_parens)]
pub(crate) mod grammar {
    // During the build step, `build.rs` will output the generated parser to `OUT_DIR` to avoid
    // adding it to the source directory, so we just directly include the generated parser here.
    //
    // Even with `.gitignore` and the `exclude` in the `Cargo.toml`, the generated parser can still
    // end up in the source directory. This could happen when `cargo build` builds the file out of
    // the Cargo cache (`$HOME/.cargo/registrysrc`), and the build script would then put its output
    // in that cached source directory because of https://github.com/lalrpop/lalrpop/issues/280.
    // Later runs of `cargo vendor` then copy the source from that directory, including the
    // generated file.
    include!(concat!(env!("OUT_DIR"), "/parser/grammar.rs"));
}

pub mod binary;
mod errors;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use firefly_intern::Symbol;
use firefly_parser::{Parse as GParse, Parser as GParser};
use firefly_parser::{Scanner, Source};
use firefly_util::diagnostics::{CodeMap, DiagnosticsHandler};

pub type Parser = GParser<ParseConfig>;
pub trait Parse<T> = GParse<T, Config = ParseConfig, Error = ParserError>;

use crate::ast;
use crate::lexer::Lexer;
use crate::preprocessor::{MacroContainer, MacroDef, MacroIdent, Preprocessed, Preprocessor};

pub use self::errors::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseConfig {
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub include_paths: VecDeque<PathBuf>,
    pub code_paths: VecDeque<PathBuf>,
    pub macros: Option<MacroContainer>,
}
impl ParseConfig {
    pub fn new() -> Self {
        ParseConfig::default()
    }

    pub fn define<V: Into<MacroDef>>(&mut self, name: Symbol, value: V) {
        let macros = self.macros.get_or_insert_with(|| MacroContainer::new());
        macros.insert(MacroIdent::Const(name), value.into());
    }
}
impl Default for ParseConfig {
    fn default() -> Self {
        ParseConfig {
            warnings_as_errors: false,
            no_warn: false,
            include_paths: VecDeque::new(),
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}

impl GParse for ast::Module {
    type Parser = grammar::ModuleParser;
    type Error = ParserError;
    type Config = ParseConfig;
    type Token = Preprocessed;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(
        parser: &Parser,
        diagnostics: &DiagnosticsHandler,
        source: S,
    ) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(parser, lexer, diagnostics);
        Self::parse_tokens(diagnostics, parser.codemap.clone(), tokens)
    }

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, Self::Error> {
        let result = Self::Parser::new().parse(diagnostics, &codemap, tokens);
        to_parse_result(result)
    }
}

impl GParse for ast::Expr {
    type Parser = grammar::ExprParser;
    type Error = ParserError;
    type Config = ParseConfig;
    type Token = Preprocessed;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(
        parser: &Parser,
        diagnostics: &DiagnosticsHandler,
        source: S,
    ) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(parser, lexer, diagnostics);
        Self::parse_tokens(diagnostics, parser.codemap.clone(), tokens)
    }

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(
        diagnostics: &DiagnosticsHandler,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, ParserError> {
        let result = Self::Parser::new().parse(diagnostics, &codemap, tokens);
        to_parse_result(result)
    }
}

fn to_parse_result<T>(result: Result<T, ParseError>) -> Result<T, ParserError> {
    match result {
        Ok(ast) => Ok(ast),
        Err(lalrpop_util::ParseError::User { error }) => Err(error.into()),
        Err(err) => Err(ParserError::from(err).into()),
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::ast::*;

    use firefly_syntax_base::BinaryOp;
    use firefly_util::diagnostics::*;

    fn parse<T, S>(config: ParseConfig, codemap: Arc<CodeMap>, input: S) -> T
    where
        T: Parse<T, Config = ParseConfig, Error = ParserError>,
        S: AsRef<str>,
    {
        let emitter = Arc::new(DefaultEmitter::new(ColorChoice::Auto));
        let diagnostics =
            DiagnosticsHandler::new(DiagnosticsConfig::default(), codemap.clone(), emitter);
        let parser = Parser::new(config, codemap);
        match parser.parse_string::<T, S, _>(&diagnostics, input) {
            Ok(ast) => return ast,
            Err(err) => {
                diagnostics.error(err);
                panic!("parsing failed");
            }
        }
    }

    fn parse_fail<T, S>(config: ParseConfig, codemap: Arc<CodeMap>, input: S) -> String
    where
        T: Parse<T, Config = ParseConfig, Error = ParserError>,
        S: AsRef<str>,
    {
        let emitter = Arc::new(CaptureEmitter::default());
        let diagnostics = DiagnosticsHandler::new(
            DiagnosticsConfig::default(),
            codemap.clone(),
            emitter.clone(),
        );
        let parser = Parser::new(config, codemap);
        match parser.parse_string::<T, S, _>(&diagnostics, input) {
            Err(err) => {
                diagnostics.error(err);
                emitter.captured()
            }
            _ => panic!("expected parse to fail, but it succeeded!"),
        }
    }

    macro_rules! module {
        ($span:expr, $codemap:expr, $name:expr, $body:expr) => {{
            let codemap = $codemap;
            let emitter = Arc::new(DefaultEmitter::new(ColorChoice::Auto));
            let diagnostics =
                DiagnosticsHandler::new(DiagnosticsConfig::default(), codemap.clone(), emitter);
            let module = Module::new_with_forms(&diagnostics, $span, $name, $body);
            if diagnostics.has_errors() {
                panic!("failed to create expected module!");
            }
            module
        }};
    }

    #[test]
    fn parse_empty_module() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(config, codemap.clone(), "-module(foo).");
        let expected = module!(result.span, &codemap, ident!(result.span, "foo"), vec![]);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_module_with_multi_clause_function() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

foo([], Acc) -> Acc;
foo([H|T], Acc) -> foo(T, [H|Acc]).
",
        );

        let span = SourceSpan::UNKNOWN;
        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(span, foo).map(Name::Atom),
            Clause {
                span,
                patterns: vec![nil!(span), var!(span, Acc)],
                guards: vec![],
                body: vec![var!(span, Acc)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(span, foo).map(Name::Atom),
            Clause {
                span,
                patterns: vec![cons!(span, var!(span, H), var!(span, T)), var!(span, Acc)],
                guards: vec![],
                body: vec![apply!(
                    span,
                    atom!(span, foo),
                    (var!(span, T), cons!(span, var!(span, H), var!(span, Acc)))
                )],
                compiler_generated: false,
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(Function {
            span,
            name: ident!(span, "foo"),
            arity: 2,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        }));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_if_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

unless(false) ->
    true;
unless(true) ->
    false;
unless(Value) ->
    if
        Value == 0 -> true;
        Value -> false;
        else -> true
    end.

",
        );

        let span = SourceSpan::UNKNOWN;
        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(span, unless).map(Name::Atom),
            Clause {
                span,
                patterns: vec![atom!(span, false)],
                guards: vec![],
                body: vec![atom!(span, true)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(span, unless).map(Name::Atom),
            Clause {
                span,
                patterns: vec![atom!(span, true)],
                guards: vec![],
                body: vec![atom!(span, false)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(span, unless).map(Name::Atom),
            Clause {
                span,
                patterns: vec![var!(span, Value)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::If(If {
                    span,
                    clauses: vec![
                        Clause::for_if(
                            span,
                            vec![Guard {
                                span,
                                conditions: vec![Expr::BinaryExpr(BinaryExpr {
                                    span,
                                    lhs: Box::new(var!(span, Value)),
                                    op: BinaryOp::Equal,
                                    rhs: Box::new(int!(span, 0.into())),
                                })],
                            }],
                            vec![atom!(span, true)],
                            false,
                        ),
                        Clause::for_if(
                            span,
                            vec![Guard {
                                span,
                                conditions: vec![var!(span, Value)],
                            }],
                            vec![atom!(span, false)],
                            false,
                        ),
                        Clause::for_if(
                            span,
                            vec![Guard {
                                span,
                                conditions: vec![atom!(span, else)],
                            }],
                            vec![atom!(span, true)],
                            false,
                        ),
                    ],
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(Function {
            span,
            name: ident!(span, unless),
            arity: 1,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        }));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_case_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

typeof(Value) ->
    case Value of
        [] -> nil;
        [_|_] -> list;
        N when is_number(N) -> N;
        _ -> other
    end.

",
        );

        let mut clauses = Vec::new();

        let span = SourceSpan::UNKNOWN;
        clauses.push((
            ident_opt!(span, typeof).map(Name::Atom),
            Clause {
                span,
                patterns: vec![var!(span, Value)],
                guards: vec![],
                body: vec![Expr::Case(Case {
                    span,
                    expr: Box::new(var!(span, Value)),
                    clauses: vec![
                        Clause {
                            span,
                            patterns: vec![nil!(span)],
                            guards: vec![],
                            body: vec![atom!(span, nil)],
                            compiler_generated: false,
                        },
                        Clause {
                            span,
                            patterns: vec![cons!(span, var!(span, _), var!(span, _))],
                            guards: vec![],
                            body: vec![atom!(span, list)],
                            compiler_generated: false,
                        },
                        Clause {
                            span,
                            patterns: vec![var!(span, N)],
                            guards: vec![Guard {
                                span,
                                conditions: vec![apply!(
                                    span,
                                    atom!(span, is_number),
                                    (var!(span, N))
                                )],
                            }],
                            body: vec![var!(span, N)],
                            compiler_generated: false,
                        },
                        Clause {
                            span,
                            patterns: vec![var!(span, _)],
                            guards: vec![],
                            body: vec![atom!(span, other)],
                            compiler_generated: false,
                        },
                    ],
                })],
                compiler_generated: false,
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(Function {
            span,
            name: ident!(span, typeof),
            arity: 1,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        }));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_receive_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

loop(State, Timeout) ->
    receive
        {From, {Ref, Msg}} ->
            From ! {Ref, ok},
            handle_info(Msg, State);
        _ ->
            exit(io_lib:format(\"unexpected message: ~p~n\", [Msg]))
    after
        Timeout ->
            timeout
    end.
",
        );

        let span = SourceSpan::UNKNOWN;
        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(span, loop).map(Name::Atom),
            Clause {
                span,
                patterns: vec![var!(span, State), var!(span, Timeout)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::Receive(Receive {
                    span,
                    clauses: Some(vec![
                        Clause {
                            span,
                            patterns: vec![tuple!(
                                span,
                                var!(span, From),
                                tuple!(span, var!(span, Ref), var!(span, Msg))
                            )],
                            guards: vec![],
                            body: vec![
                                Expr::BinaryExpr(BinaryExpr {
                                    span,
                                    lhs: Box::new(var!(span, From)),
                                    op: BinaryOp::Send,
                                    rhs: Box::new(tuple!(span, var!(span, Ref), atom!(span, ok))),
                                }),
                                apply!(
                                    span,
                                    atom!(span, handle_info),
                                    (var!(span, Msg), var!(span, State))
                                ),
                            ],
                            compiler_generated: false,
                        },
                        Clause {
                            span,
                            patterns: vec![var!(span, _)],
                            guards: vec![],
                            body: vec![apply!(
                                span,
                                atom!(span, exit),
                                (apply!(
                                    span,
                                    io_lib,
                                    format,
                                    (
                                        Expr::Literal(Literal::String(ident!(
                                            span,
                                            "unexpected message: ~p~n"
                                        ))),
                                        cons!(span, var!(span, Msg), nil!(span))
                                    )
                                ))
                            )],
                            compiler_generated: false,
                        },
                    ]),
                    after: Some(After {
                        span,
                        timeout: Box::new(var!(span, Timeout)),
                        body: vec![atom!(span, timeout)],
                    }),
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(Function {
            span,
            name: ident!(span, loop),
            arity: 2,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        }));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_if() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).
-define(TEST, true).
-define(OTP_VERSION, 21).

-ifdef(TEST).
env() ->
    test.
-else.
env() ->
    release.
-endif.

-if(?OTP_VERSION > 21).
system_version() ->
    future.
-elif(?OTP_VERSION == 21).
system_version() ->
    ?OTP_VERSION.
-else.
system_version() ->
    old.
-endif.
",
        );

        let span = SourceSpan::UNKNOWN;
        let mut body = Vec::new();
        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(span, env).map(Name::Atom),
            Clause {
                span,
                patterns: vec![],
                guards: vec![],
                body: vec![atom!(span, test)],
                compiler_generated: false,
            },
        ));
        let env_fun = Function {
            span,
            name: ident!(span, env),
            arity: 0,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        };
        body.push(TopLevel::Function(env_fun));

        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(span, system_version).map(Name::Atom),
            Clause {
                span,
                patterns: vec![],
                guards: vec![],
                body: vec![int!(span, 21.into())],
                compiler_generated: false,
            },
        ));
        let system_version_fun = Function {
            span,
            name: ident!(span, system_version),
            arity: 0,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        };
        body.push(TopLevel::Function(system_version_fun));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_warning_error() {
        // NOTE: Warnings are not printed with cfg(test), as we
        // cannot control where they end up without refactoring to pass
        // a writer everywhere. You can change this for testing by
        // going to the Preprocessor and finding the line where we handle
        // the warning directive and toggle the config flag
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let captured = parse_fail::<Module, &str>(
            config,
            codemap.clone(),
            "-module(foo).
-warning(\"this is a compiler warning\").
-error(\"this is a compiler error\").
",
        );
        assert!(captured.contains("this is a compiler warning"));
        assert!(captured.contains("this is a compiler error"));
    }

    #[test]
    fn parse_try() {
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

example(File) ->
    try read(File) of
        {ok, Contents} ->
            {ok, Contents}
    catch
        error:{Mod, Code} ->
            {error, Mod:format_error(Code)};
        Reason ->
            {error, Reason}
    after
        close(File)
    end.
",
        );

        let span = SourceSpan::UNKNOWN;
        let mut clauses = Vec::new();

        let callee = Expr::Remote(Remote::new(
            span,
            var!(span, Mod),
            atom!(span, format_error),
        ));
        clauses.push((
            ident_opt!(span, example).map(Name::Atom),
            Clause {
                span,
                patterns: vec![var!(span, File)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::Try(Try {
                    span,
                    exprs: vec![apply!(span, atom!(span, read), (var!(span, File)))],
                    clauses: Some(vec![Clause {
                        span,
                        patterns: vec![tuple!(span, atom!(span, ok), var!(span, Contents))],
                        guards: vec![],
                        body: vec![tuple!(span, atom!(span, ok), var!(span, Contents))],
                        compiler_generated: false,
                    }]),
                    catch_clauses: Some(vec![
                        Clause::for_catch(
                            span,
                            atom!(span, error),
                            tuple!(span, var!(span, Mod), var!(span, Code)),
                            Some(var!(span, _)),
                            vec![],
                            vec![tuple!(
                                span,
                                atom!(span, error),
                                apply!(span, callee, (var!(span, Code)))
                            )],
                        ),
                        Clause::for_catch(
                            span,
                            atom!(span, throw),
                            var!(span, Reason),
                            Some(var!(span, _)),
                            vec![],
                            vec![tuple!(span, atom!(span, error), var!(span, Reason))],
                        ),
                    ]),
                    after: Some(vec![apply!(span, atom!(span, close), (var!(span, File)))]),
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(Function {
            span,
            name: ident!(span, example),
            arity: 1,
            spec: None,
            is_nif: false,
            clauses,
            var_counter: 0,
            fun_counter: 0,
        }));
        let expected = module!(span, &codemap, ident!(span, foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_try2() {
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let _result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

example(File < 2) ->
    try read(File) of
        {ok, Contents} ->
            {ok, Contents}
    catch
        error:{Mod, Code} ->
            {error, Mod:format_error(Code)};
        Reason ->
            {error, Reason}
    after
        close(File)
    end.

exw(File) ->
    case File of
        File < 2 ->
            ok
    end.
",
        );
    }

    #[test]
    fn parse_numbers() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

foo(F) -> F-1+1/1*1.

bar() -> - 2.
",
        );
    }

    #[test]
    fn parse_spec() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

-spec bar() -> number.
bar() -> 2.
",
        );
    }

    #[test]
    fn parse_binary_spec_constant() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

-type txs_hash() :: <<_:(32 * 8)>>.
-type a() :: <<_:A * (12 * 8)>>.
",
        );
    }

    #[test]
    fn parse_multi_line_string() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-deprecated([{woo, 0,
"testing testing"
"testing testing"}]).
woo() -> 2.
"#,
        );
    }

    #[test]
    fn parse_multi_line_record() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-record(some_record,
        {
            woohoo
        }).
"#,
        );
    }

    #[test]
    fn parse_removed_attribute() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-removed([{foo, 0, "we don't support this anymore!"}]).
"#,
        );
    }

    #[test]
    fn parse_on_load() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).
-on_load(bar/0).

bar() -> yay.
"#,
        );
    }

    #[test]
    #[ignore]
    fn parse_elixir_enum_erl() {
        use std::io::Read;

        let file = std::fs::File::open("../test_data/Elixir.Enum.erl");
        let mut string = String::new();
        file.unwrap().read_to_string(&mut string).unwrap();

        let codemap = Arc::new(CodeMap::new());
        let _result: Module = parse(ParseConfig::default(), codemap, &string);
    }
}
