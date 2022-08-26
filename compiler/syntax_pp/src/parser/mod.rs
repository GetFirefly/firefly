mod error;
pub use self::error::ParseError;

mod lexer;
pub use self::lexer::Lexer;

mod token;
pub use self::token::Token;

/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => {
        firefly_diagnostics::SourceSpan::new($l, $r)
    };
    ($i:expr) => {
        firefly_diagnostics::SourceSpan::new($i, $i)
    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
#[allow(unused)]
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
    include!(concat!(env!("OUT_DIR"), "/grammar.rs"));
}

use firefly_beam::beam::AbstractCode;
use firefly_diagnostics::{ToDiagnostic, Diagnostic, Label, Reporter, SourceSpan};
use firefly_parser::{Parse, Parser, Scanner, Source};

use crate::ast::{self, Form};

/// Parses forms from the given AbstractCode object
pub from_abstract_code(code: &AbstractCode) -> anyhow::Result<Form> {
    use firefly_beam::serialization::etf::pattern::VarList;

    let (_, form) = code
        .code()
        .as_match(("raw_abstract_v1", VarList(FormPattern)))?;

    Ok(form)
}

impl Parse for ast::Root {
    type Parser = grammar::RootParser;
    type Error = ParseError;
    type Config = ();
    type Token = Result<(SourceIndex, Token, SourceIndex), ()>;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParseError::RootFileError { source, path }
    }

    fn parse<S>(
        _parser: &Parser<Self::Config>,
        reporter: Reporter,
        source: S,
    ) -> Result<Self, ()>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        Self::parse_tokens(errors, lexer)
    }

    fn parse_tokens<S>(reporter: Reporter, tokens: S) -> Result<Self, ()>
    where
        S: IntoIterator<Item = Self::Token>,
    {
        match Self::Parser::new().parse(tokens) {
            Ok(inner) => Ok(inner),
            Err(err) => {
                errors.error(err.into());
                Err(())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use firefly_diagnostics::*;
    use firefly_parser::{Parse, Parser};

    use super::*;

    fn fail_with(reporter: &Reporter, codemap: &CodeMap) -> ! {
        use term::termcolor::{ColorChoice, StandardStream};

        let config = term::Config::default();
        let mut out = StandardStream::stderr(ColorChoice::Always);
        for diagnostic in reporter.diagnostics() {
            term::emit(&mut out, &config, codemap, &diagnostic).unwrap();
        }
        panic!();
    }

    fn parse<T, S>(input: S) -> T
    where
        T: Parse<T, Config = (), Error = ParseError>,
        S: AsRef<str>,
    {
        let parser = Parser::new((), Arc::new(CodeMap::new()));
        let mut errors = Errors::new();
        match parser.parse_string::<T, S>(&mut errors, input) {
            Ok(ast) => return ast,
            Err(()) => fail_with(&errors, &parser.codemap),
        };
    }

    #[test]
    fn simple() {
        let _: Root = parse(
            "
{woo, '123fwoo', {}}.
",
        );
    }

    #[test]
    fn basic_ast() {
        let _: Root = parse(
            "
{attribute,1,file,{\"woo.erl\",1}}.
{attribute,1,module,woo}.
{attribute,3,export,[{foo,2},{bar,1},{barr,1}]}.
{function,5,foo,2,
    [{clause,5,
    [{var,5,'A'},{var,5,'B'}],
    [],
    [{op,5,'+',{var,5,'A'},{var,5,'B'}}]}]}.
{function,7,bar,1,
    [{clause,7,[{integer,7,1}],[],[{integer,7,2}]},
    {clause,8,[{integer,8,2}],[],[{integer,8,4}]},
    {clause,9,[{var,9,'N'}],[],[{var,9,'N'}]}]}.
{function,11,barr,1,
    [{clause,11,[{integer,11,1}],[],[{integer,11,2}]},
    {clause,12,[{integer,12,2}],[],[{integer,12,4}]}]}.
{function,14,binary,0,
    [{clause,14,[],[],
    [{bin,14,[{bin_element,14,{string,14,\"woo\"},default,default}]}]}]}.
{function,16,string,0,[{clause,16,[],[],[{string,16,\"woo\"}]}]}.
{eof,17}.
",
        );
    }
}
