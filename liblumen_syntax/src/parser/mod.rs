/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => (ByteSpan::new($l, $r));
    ($i:expr) => (ByteSpan::new($i, $i))
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
mod grammar {
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

pub mod ast;
mod errors;
/// Contains the visitor trait needed to traverse the AST and helper walk functions.
//pub mod visitor;

use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};

use liblumen_diagnostics::{CodeMap, FileName};

use crate::lexer::{Lexer, Scanner, Source, FileMapSource, Symbol};
use crate::preprocessor::{Preprocessor, Preprocessed, MacroDef};

pub use self::errors::*;

/// The type of result returned from parsing functions
pub type ParseResult<T> = Result<T, Vec<ParserError>>;

pub struct Parser {
    pub config: ParseConfig,
}
impl Parser {
    pub fn new(config: ParseConfig) -> Parser {
        Parser {
            config,
        }
    }

    pub fn parse_string<S, T>(&self, source: S) -> ParseResult<T>
    where
        S: AsRef<str>,
        T: Parse,
    {
        let filemap = {
            self.config.codemap.lock()
                .unwrap()
                .add_filemap(FileName::Virtual(Cow::Borrowed("nofile")), source.as_ref().to_owned())
        };
        <T as Parse<T>>::parse(&self.config, FileMapSource::new(filemap))
    }

    pub fn parse_file<P, T>(&self, path: P) -> ParseResult<T>
    where
        P: AsRef<Path>,
        T: Parse,
    {
        match FileMapSource::from_path(self.config.codemap.clone(), path) {
            Err(err) =>
                return Err(vec![err.into()]),
            Ok(source) =>
                <T as Parse<T>>::parse(&self.config, source)
        }
    }
}

pub struct ParseConfig {
    pub codemap: Arc<Mutex<CodeMap>>,
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub code_paths: VecDeque<PathBuf>,
    pub macros: Option<HashMap<Symbol, MacroDef>>,
}
impl ParseConfig {
    pub fn new(codemap: Arc<Mutex<CodeMap>>) -> Self {
        ParseConfig {
            codemap,
            warnings_as_errors: false,
            no_warn: false,
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}
impl Default for ParseConfig {
    fn default() -> Self {
        ParseConfig {
            codemap: Arc::new(Mutex::new(CodeMap::new())),
            warnings_as_errors: false,
            no_warn: false,
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}

pub trait Parse<T = Self> {
    type Parser;

    /// Initializes a token stream for the underlying parser and invokes parse_tokens/1
    fn parse<S>(config: &ParseConfig, source: S) -> ParseResult<T>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(config, lexer);
        Self::parse_tokens(tokens)
    }

    /// Implemented by each parser, which should parse the token stream and produce a T
    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<T>;
}

impl Parse for ast::Module {
    type Parser = grammar::ModuleParser;

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<ast::Module> {
        let mut errs = Vec::new();
        let result = Self::Parser::new()
            .parse(&mut errs, tokens)
            .map_err(|e| e.map_error(|ei| ei.into()));
        to_parse_result(errs, result)
    }
}

impl Parse for ast::Expr {
    type Parser = grammar::ExprParser;

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<ast::Expr> {
        let mut errs = Vec::new();
        let result = Self::Parser::new()
            .parse(&mut errs, tokens)
            .map_err(|e| e.map_error(|ei| ei.into()));
        to_parse_result(errs, result)
    }
}


fn to_parse_result<T>(mut errs: Vec<ParseError>, result: Result<T, ParseError>) -> ParseResult<T> {
    match result {
        Ok(ast) => {
            if errs.len() > 0 {
                return Err(errs.drain(0..).map(ParserError::from).collect());
            }
            Ok(ast)
        }
        Err(err) => {
            errs.push(err);
            Err(errs.drain(0..).map(ParserError::from).collect())
        }
    }
}


#[cfg(test)]
mod test {
    use termcolor::ColorChoice;

    use pretty_assertions::assert_eq;

    use super::*;
    use super::ast::*;

    use liblumen_diagnostics::{ByteSpan, ByteIndex};
    use liblumen_diagnostics::{Emitter, StandardStreamEmitter};

    use crate::preprocessor::PreprocessorError;
    use crate::lexer::{Ident, Symbol};

    macro_rules! span_usize {
        ($l:expr, $r:expr) => (ByteSpan::new(ByteIndex($l), ByteIndex($r)));
        ($i:expr) => (ByteSpan::new(ByteIndex($i), ByteIndex($i)));
    }

    macro_rules! ident {
        ($sym:expr) => (Ident::new(Symbol::intern($sym), span_usize!(1, 1)));
        ($sym:expr, $l:expr) => (Ident::new(Symbol::intern($sym), span_usize!($l, $l)));
        ($sym:expr, $l:expr, $r:expr) => (Ident::new(Symbol::intern($sym), span_usize!($l, $r)));
    }

    fn parse<T>(input: &'static str) -> T
    where
        T: Parse<T>,
    {
        let config = ParseConfig::default();
        let parser = Parser::new(config);
        let errs = match parser.parse_string::<&'static str, T>(input) {
            Ok(ast) => return ast,
            Err(errs) => errs
        };
        let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
            .set_codemap(parser.config.codemap.clone());
        for err in errs.iter() {
            emitter.diagnostic(&err.to_diagnostic()).unwrap();
        }
        panic!("parse failed");
    }

    fn parse_fail<T>(input: &'static str) -> Vec<ParserError>
    where
        T: Parse<T>,
    {
        let config = ParseConfig::default();
        let parser = Parser::new(config);
        match parser.parse_string::<&'static str, T>(input) {
            Err(errs) => errs,
            _ => panic!("expected parse to fail, but it succeeded!"),
        }
    }


    macro_rules! parser_test {
        ($name:ident, $blk:block) => {
            #[test]
            fn $name() {
                // TODO: Clean this up, no longer need these macros
                $blk
            }
        }
    }

    parser_test!(parse_empty_module, {
        let result: Module = parse("-module(foo).");
        let expected = Module {
            span: span_usize!(1, 14),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions: Vec::new(),
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_module_with_multi_clause_function, {
        let result: Module = parse("-module(foo).

foo([], Acc) -> Acc;
foo([H|T], Acc) -> foo(T, [H|Acc]).
");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(16, 35),
            name: Some(ident!("foo", 16, 19)),
            params: vec![
                Pattern::Nil(span_usize!(20, 22)),
                Pattern::Var(ident!("Acc", 24, 27))
            ],
            guard: None,
            body: vec![
                Expr::Var(ident!("Acc", 32, 35))
            ]
        });
        clauses.push(FunctionClause {
            span: span_usize!(37, 71),
            name: Some(ident!("foo", 37, 40)),
            params: vec![
                Pattern::Cons(span_usize!(41, 46),
                    Box::new(Pattern::Var(ident!("H", 42, 43))),
                    Box::new(Pattern::Var(ident!("T", 44, 45)))
                ),
                Pattern::Var(ident!("Acc", 48, 51))
            ],
            guard: None,
            body: vec![
                Expr::Apply{
                    span: span_usize!(56, 71),
                    lhs: Box::new(Expr::Literal(Literal::Atom(ident!("foo", 56, 59)))),
                    args: vec![
                        Expr::Var(ident!("T", 60, 61)),
                        Expr::Cons(
                            span_usize!(63, 70),
                            Box::new(Expr::Var(ident!("H", 64, 65))),
                            Box::new(Expr::Var(ident!("Acc", 66, 69)))
                        )
                    ]
                }
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(16, 72),
            name: ident!("foo", 16, 19),
            arity: 2,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 72),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_if_expressions, {
        let result: Module = parse("-module(foo).

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

");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(16, 41),
            name: Some(ident!("unless", 16, 22)),
            params: vec![
                Pattern::Literal(Literal::Atom(ident!("false", 23, 28)))
            ],
            guard: None,
            body: vec![
                Expr::Literal(Literal::Atom(ident!("true", 37, 41)))
            ]
        });
        clauses.push(FunctionClause {
            span: span_usize!(43, 68),
            name: Some(ident!("unless", 43, 49)),
            params: vec![
                Pattern::Literal(Literal::Atom(ident!("true", 50, 54)))
            ],
            guard: None,
            body: vec![
                Expr::Literal(Literal::Atom(ident!("false", 63, 68)))
            ]
        });
        clauses.push(FunctionClause {
            span: span_usize!(70, 174),
            name: Some(ident!("unless", 70, 76)),
            params: vec![
                Pattern::Var(ident!("Value", 77, 82)),
            ],
            guard: None,
            body: vec![
                Expr::If(span_usize!(91, 174), vec![
                    IfClause(
                        span_usize!(102, 120),
                        vec![
                            Expr::BinaryExpr {
                                span: span_usize!(102, 112),
                                lhs: Box::new(Expr::Var(ident!("Value", 102, 107))),
                                op: BinaryOp::Equal,
                                rhs: Box::new(Expr::Literal(Literal::Integer(span_usize!(111, 112), 0))),
                            }
                        ],
                        vec![
                            Expr::Literal(Literal::Atom(ident!("true", 116, 120))),
                        ]
                    ),
                    IfClause(
                        span_usize!(130, 144),
                        vec![Expr::Var(ident!("Value", 130, 135))],
                        vec![
                            Expr::Literal(Literal::Atom(ident!("false", 139, 144))),
                        ]
                    ),
                    IfClause(
                        span_usize!(154, 166),
                        vec![Expr::Literal(Literal::Atom(ident!("else", 154, 158)))],
                        vec![
                            Expr::Literal(Literal::Atom(ident!("true", 162, 166))),
                        ]
                    )
                ])
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(16, 175),
            name: ident!("unless", 16, 22),
            arity: 1,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 175),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_case_expressions, {
        let result: Module = parse("-module(foo).

typeof(Value) ->
    case Value of
        [] -> nil;
        [_|_] -> list;
        N when is_number(N) -> N;
        _ -> other
    end.

");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(16, 153),
            name: Some(ident!("typeof", 16, 22)),
            params: vec![
                Pattern::Var(ident!("Value", 23, 28)),
            ],
            guard: None,
            body: vec![
                Expr::Case(
                    span_usize!(37, 153),
                    Box::new(Expr::Var(ident!("Value", 42, 47))),
                    vec![
                        Clause {
                            span: span_usize!(59, 68),
                            pattern: Pattern::Nil(span_usize!(59, 61)),
                            guard: None,
                            body: vec![Expr::Literal(Literal::Atom(ident!("nil", 65, 68)))],
                        },
                        Clause {
                            span: span_usize!(78, 91),
                            pattern: Pattern::Cons(
                                span_usize!(78, 83),
                                Box::new(Pattern::Var(ident!("_", 79, 80))),
                                Box::new(Pattern::Var(ident!("_", 81, 82))),
                            ),
                            guard: None,
                            body: vec![Expr::Literal(Literal::Atom(ident!("list", 87, 91)))],
                        },
                        Clause {
                            span: span_usize!(101, 125),
                            pattern: Pattern::Var(ident!("N", 101, 102)),
                            guard: Some(vec![
                                Guard {
                                    span: span_usize!(108, 120),
                                    conditions: vec![
                                        Expr::Apply {
                                            span: span_usize!(108, 120),
                                            lhs: Box::new(Expr::Literal(Literal::Atom(ident!("is_number", 108, 117)))),
                                            args: vec![
                                                Expr::Var(ident!("N", 118, 119)),
                                            ]
                                        }
                                    ]
                                }
                            ]),
                            body: vec![Expr::Var(ident!("N", 124, 125))],
                        },
                        Clause {
                            span: span_usize!(135, 145),
                            pattern: Pattern::Var(ident!("_", 135, 136)),
                            guard: None,
                            body: vec![Expr::Literal(Literal::Atom(ident!("other", 140, 145)))],
                        }
                    ]
                )
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(16, 154),
            name: ident!("typeof", 16, 22),
            arity: 1,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 154),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_receive_expressions, {
        let result: Module = parse("-module(foo).

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
");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(16, 285),
            name: Some(ident!("loop", 16, 20)),
            params: vec![
                Pattern::Var(ident!("State", 21, 26)),
                Pattern::Var(ident!("Timeout", 28, 35))
            ],
            guard: None,
            body: vec![
                Expr::Receive {
                    span: span_usize!(44, 285),
                    clauses: Some(vec![
                        Clause {
                            span: span_usize!(60, 147),
                            pattern: Pattern::Tuple(span_usize!(60, 78), vec![
                                Expr::Var(ident!("From", 61, 65)),
                                Expr::Tuple(span_usize!(67, 77), vec![
                                    Expr::Var(ident!("Ref", 68, 71)),
                                    Expr::Var(ident!("Msg", 73, 76)),
                                ])
                            ]),
                            guard: None,
                            body: vec![
                                Expr::BinaryExpr {
                                    span: span_usize!(94, 110),
                                    lhs: Box::new(Expr::Var(ident!("From", 94, 98))),
                                    op: BinaryOp::Send,
                                    rhs: Box::new(Expr::Tuple(span_usize!(101, 110), vec![
                                        Expr::Var(ident!("Ref", 102, 105)),
                                        Expr::Literal(Literal::Atom(ident!("ok", 107, 109))),
                                    ]))
                                },
                                Expr::Apply {
                                    span: span_usize!(124, 147),
                                    lhs: Box::new(Expr::Literal(Literal::Atom(ident!("handle_info", 124, 135)))),
                                    args: vec![
                                        Expr::Var(ident!("Msg", 136, 139)),
                                        Expr::Var(ident!("State", 141, 146)),
                                    ]
                                }
                            ]
                        },
                        Clause {
                            span: span_usize!(157, 228),
                            pattern: Pattern::Var(ident!("_", 157, 158)),
                            guard: None,
                            body: vec![
                                Expr::Apply {
                                    span: span_usize!(174, 228),
                                    lhs: Box::new(Expr::Literal(Literal::Atom(ident!("exit", 174, 178)))),
                                    args: vec![
                                        Expr::Apply {
                                            span: span_usize!(179, 227),
                                            lhs: Box::new(Expr::Remote {
                                                span: span_usize!(179, 192),
                                                module: Box::new(Expr::Literal(Literal::Atom(ident!("io_lib", 179, 185)))),
                                                function: Box::new(Expr::Literal(Literal::Atom(ident!("format", 186, 192)))),
                                            }),
                                            args: vec![
                                                Expr::Literal(Literal::String(ident!("unexpected message: ~p~n", 193, 219))),
                                                Expr::Cons(
                                                    span_usize!(221, 226),
                                                    Box::new(Expr::Var(ident!("Msg", 222, 225))),
                                                    Box::new(Expr::Nil(span_usize!(225, 226)))
                                                )
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]),
                    after: Some(Timeout(
                        span_usize!(233, 277),
                        Box::new(Expr::Var(ident!("Timeout", 247, 254))),
                        vec![Expr::Literal(Literal::Atom(ident!("timeout", 270, 277)))]
                    ))
                }
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(16, 286),
            name: ident!("loop", 16, 20),
            arity: 2,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 286),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_preprocessor_if, {
        let result: Module = parse("-module(foo).
-define(TEST, true).

-ifdef(TEST).
env() ->
    test.
-else.
env() ->
    release.
-endif.
");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(51, 68),
            name: Some(ident!("env", 51, 54)),
            params: vec![],
            guard: None,
            body: vec![
                Expr::Literal(Literal::Atom(ident!("test", 64, 68)))
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(51, 69),
            name: ident!("env", 51, 54),
            arity: 0,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 69),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    parser_test!(parse_preprocessor_warning_error, {
        // NOTE: Warnings are not printed with cfg(test), as we
        // cannot control where they end up without refactoring to pass
        // a writer everywhere. You can change this for testing by
        // going to the Preprocessor and finding the line where we handle
        // the warning directive and toggle the config flag
        let mut errs = parse_fail::<Module>("-module(foo).
-warning(\"this is a compiler warning\").
-error(\"this is a compiler error\").
");
        match errs.pop() {
            Some(ParserError::Preprocessor(PreprocessorError::CompilerError(_, _))) => (),
            Some(err) => panic!("expected compiler error, but got a different error instead: {:?}", err),
            None => panic!("expected compiler error, but didn't get any errors!"),
        }
    });

    parser_test!(parse_try, {
        let result: Module = parse("-module(foo).

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
");
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: span_usize!(16, 275),
            name: Some(ident!("example", 16, 23)),
            params: vec![
                Pattern::Var(ident!("File", 24, 28)),
            ],
            guard: None,
            body: vec![
                Expr::Try {
                    span: span_usize!(37, 275),
                    exprs: Some(vec![
                        Expr::Apply {
                            span: span_usize!(41, 51),
                            lhs: Box::new(Expr::Literal(Literal::Atom(ident!("read", 41, 45)))),
                            args: vec![ Expr::Var(ident!("File", 46, 50)) ]
                        },
                    ]),
                    clauses: Some(vec![
                        Clause {
                            span: span_usize!(63, 107),
                            pattern: Pattern::Tuple(span_usize!(63, 77), vec![
                                Expr::Literal(Literal::Atom(ident!("ok", 64, 66))),
                                Expr::Var(ident!("Contents", 68, 76)),
                            ]),
                            guard: None,
                            body: vec![
                                Expr::Tuple(span_usize!(93, 107), vec![
                                    Expr::Literal(Literal::Atom(ident!("ok", 94, 96))),
                                    Expr::Var(ident!("Contents", 98, 106)),
                                ]),
                            ]
                        },
                    ]),
                    catch_clauses: Some(vec![
                        TryClause {
                            span: span_usize!(126, 190),
                            kind: ident!("error", 126, 131),
                            error: Pattern::Tuple(span_usize!(132, 143), vec![
                                Expr::Var(ident!("Mod", 133, 136)),
                                Expr::Var(ident!("Code", 138, 142)),
                            ]),
                            trace: ident!("_", 0, 0),
                            guard: None,
                            body: vec![
                                Expr::Tuple(span_usize!(159, 190), vec![
                                    Expr::Literal(Literal::Atom(ident!("error", 160, 165))),
                                    Expr::Apply {
                                        span: span_usize!(167, 189),
                                        lhs: Box::new(Expr::Remote {
                                            span: span_usize!(167, 183),
                                            module: Box::new(Expr::Var(ident!("Mod", 167, 170))),
                                            function: Box::new(Expr::Literal(Literal::Atom(ident!("format_error", 171, 183)))),
                                        }),
                                        args: vec![ Expr::Var(ident!("Code", 184, 188)) ],
                                    }
                                ])
                            ]
                        },
                        TryClause {
                            span: span_usize!(200, 237),
                            kind: ident!("throw", 0, 0),
                            error: Pattern::Var(ident!("Reason", 200, 206)),
                            trace: ident!("_", 0, 0),
                            guard: None,
                            body: vec![
                                Expr::Tuple(span_usize!(222, 237), vec![
                                    Expr::Literal(Literal::Atom(ident!("error", 223, 228))),
                                    Expr::Var(ident!("Reason", 230, 236)),
                                ])
                            ]
                        }
                    ]),
                    after: Some(vec![
                        Expr::Apply {
                            span: span_usize!(256, 267),
                            lhs: Box::new(Expr::Literal(Literal::Atom(ident!("close", 256, 261)))),
                            args: vec![ Expr::Var(ident!("File", 262, 266)) ],
                        }
                    ])
                }
            ]
        });
        let mut functions = Vec::new();
        functions.push(Function {
            span: span_usize!(16, 276),
            name: ident!("example", 16, 23),
            arity: 1,
            clauses
        });
        let expected = Module {
            span: span_usize!(1, 276),
            name: ident!("foo", 9, 12),
            attributes: Vec::new(),
            records: Vec::new(),
            functions,
        };
        assert_eq!(result, expected);
    });

    // TODO: Add support for:
    // Map/Record expressions, and any other unimplemented constructs
}
