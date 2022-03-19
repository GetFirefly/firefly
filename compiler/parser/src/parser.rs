use std::path::{Path, PathBuf};
use std::sync::Arc;

use liblumen_diagnostics::*;

use crate::{FileMapSource, Source};

pub struct Parser<C> {
    pub config: C,
    pub codemap: Arc<CodeMap>,
}

impl<C> Parser<C> {
    pub fn new(config: C, codemap: Arc<CodeMap>) -> Self {
        Self { config, codemap }
    }
}

impl<C> Parser<C> {
    pub fn parse<T>(&self, reporter: Reporter, source: Arc<SourceFile>) -> Result<T, ()>
    where
        T: Parse<Config = C>,
    {
        <T as Parse<T>>::parse(&self, reporter, FileMapSource::new(source))
    }

    pub fn parse_string<T, S>(&self, reporter: Reporter, source: S) -> Result<T, ()>
    where
        T: Parse<Config = C>,
        S: AsRef<str>,
    {
        let id = self.codemap.add("nofile", source.as_ref().to_string());
        let file = self.codemap.get(id).unwrap();
        self.parse(reporter, file)
    }

    pub fn parse_file<T, S>(&self, reporter: Reporter, source: S) -> Result<T, ()>
    where
        T: Parse<Config = C>,
        S: AsRef<Path>,
    {
        let path = source.as_ref();
        match std::fs::read_to_string(path) {
            Err(err) => {
                reporter.error(<T as Parse<T>>::root_file_error(err, path.to_owned()));
                Err(())
            }
            Ok(content) => {
                let id = self.codemap.add(path, content);
                let file = self.codemap.get(id).unwrap();
                self.parse(reporter, file)
            }
        }
    }
}

pub trait Parse<T = Self> {
    type Parser;
    type Error: ToDiagnostic;
    type Config;
    type Token;

    fn root_file_error(err: std::io::Error, path: PathBuf) -> Self::Error;

    /// Initializes a token stream for the underlying parser and invokes parse_tokens
    fn parse<S>(parser: &Parser<Self::Config>, reporter: Reporter, source: S) -> Result<T, ()>
    where
        S: Source;

    /// Implemented by each parser, which should parse the token stream and produce a T
    fn parse_tokens<S>(reporter: Reporter, tokens: S) -> Result<T, ()>
    where
        S: IntoIterator<Item = Self::Token>;
}
