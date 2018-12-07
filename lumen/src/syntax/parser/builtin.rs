use trackable::track;

use crate::syntax::preprocessor;
use crate::syntax::preprocessor::Preprocessor;
use crate::syntax::tokenizer::{Lexer, LexicalToken};

use super::cst::ModuleDecl;
use super::traits::TokenRead;
use super::{Parser, Result, TokenReader};

#[derive(Debug)]
pub struct ModuleParser<'a>(Parser<TokenReader<Preprocessor<Lexer<&'a str>>, preprocessor::Error>>);
impl<'a> ModuleParser<'a> {
    pub fn new(tokens: Preprocessor<Lexer<&'a str>>) -> Self {
        ModuleParser(Parser::new(TokenReader::new(tokens)))
    }
    pub fn parse_module(&mut self) -> Result<ModuleDecl> {
        track!(self.0.parse(), "next={:?}", self.0.parse::<LexicalToken>())
    }
    pub fn preprocessor(&self) -> &Preprocessor<Lexer<&'a str>> {
        self.0.reader().inner()
    }
    pub fn preprocessor_mut(&mut self) -> &mut Preprocessor<Lexer<&'a str>> {
        self.0.reader_mut().inner_mut()
    }
}

pub fn parse_module(tokens: &mut TokenRead) -> Result<ModuleDecl> {
    Parser::new(tokens).parse()
}
