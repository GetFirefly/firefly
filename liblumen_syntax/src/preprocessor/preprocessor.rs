use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use failure::{format_err, Error};
use termcolor::ColorChoice;

use liblumen_diagnostics::{ByteIndex, ByteSpan, CodeMap, Diagnostic, Label};
use liblumen_diagnostics::{Emitter, StandardStreamEmitter};

use crate::lexer::{symbols, IdentToken, Lexed, LexicalToken, Symbol, Token};
use crate::lexer::{Lexer, Source};
use crate::parser::ParseConfig;

use super::macros::Stringify;
use super::token_reader::{TokenBufferReader, TokenReader, TokenStreamReader};
use super::{Directive, MacroCall, MacroDef};
use super::{Preprocessed, PreprocessorError, Result};

pub struct Preprocessor<Reader: TokenReader> {
    codemap: Arc<Mutex<CodeMap>>,
    reader: Reader,
    can_directive_start: bool,
    directives: BTreeMap<ByteIndex, Directive>,
    code_paths: VecDeque<PathBuf>,
    branches: Vec<Branch>,
    macros: HashMap<Symbol, MacroDef>,
    macro_calls: BTreeMap<ByteIndex, MacroCall>,
    expanded_tokens: VecDeque<LexicalToken>,
    warnings_as_errors: bool,
    no_warn: bool,
}
impl<S> Preprocessor<TokenStreamReader<S>>
where
    S: Source,
{
    pub fn new(config: &ParseConfig, tokens: Lexer<S>) -> Self {
        let codemap = config.codemap.clone();
        let reader = TokenStreamReader::new(codemap.clone(), tokens);
        let code_paths = config.code_paths.clone();
        let macros = match config.macros {
            None => HashMap::new(),
            Some(ref macros) => macros.clone(),
        };
        Preprocessor {
            codemap,
            reader,
            can_directive_start: true,
            directives: BTreeMap::new(),
            code_paths,
            branches: Vec::new(),
            macros,
            macro_calls: BTreeMap::new(),
            expanded_tokens: VecDeque::new(),
            warnings_as_errors: config.warnings_as_errors,
            no_warn: config.no_warn,
        }
    }
}
impl<R, S> Preprocessor<R>
where
    R: TokenReader<Source = S>,
{
    fn clone_with(&self, tokens: VecDeque<Lexed>) -> Preprocessor<TokenBufferReader> {
        let codemap = self.codemap.clone();
        let reader = TokenBufferReader::new(codemap.clone(), tokens);
        Preprocessor {
            codemap,
            reader,
            can_directive_start: false,
            directives: BTreeMap::new(),
            code_paths: self.code_paths.clone(),
            branches: Vec::new(),
            macros: self.macros.clone(),
            macro_calls: BTreeMap::new(),
            expanded_tokens: VecDeque::new(),
            warnings_as_errors: self.warnings_as_errors,
            no_warn: self.no_warn,
        }
    }

    fn ignore(&self) -> bool {
        self.branches.iter().any(|b| !b.entered)
    }

    fn next_token(&mut self) -> Result<Option<LexicalToken>> {
        loop {
            if let Some(token) = self.expanded_tokens.pop_front() {
                return Ok(Some(token));
            }
            if self.can_directive_start {
                match self.try_read_directive()? {
                    Some(Directive::Module(d)) => {
                        // We need to expand this directive back to a token stream for the parser
                        self.expanded_tokens = d.expand();
                        // Otherwise treat it like other directives
                        self.directives
                            .insert(d.span().start(), Directive::Module(d));
                        continue;
                    }
                    Some(d) => {
                        self.directives.insert(d.span().start(), d);
                        continue;
                    }
                    None => (),
                }
            }
            if !self.ignore() {
                if let Some(m) = self.reader.try_read_macro_call(&self.macros)? {
                    self.macro_calls.insert(m.span().start(), m.clone());
                    self.expanded_tokens = self.expand_macro(m)?;
                    continue;
                }
            }
            if let Some(token) = self.reader.try_read_token()? {
                if self.ignore() {
                    continue;
                }
                if let LexicalToken(_, Token::Dot, _) = token {
                    self.can_directive_start = true;
                } else {
                    self.can_directive_start = false;
                }
                return Ok(Some(token));
            } else {
                break;
            }
        }
        Ok(None)
    }

    fn expand_macro(&self, call: MacroCall) -> Result<VecDeque<LexicalToken>> {
        if let Some(expanded) = self.try_expand_predefined_macro(&call)? {
            Ok(vec![expanded].into())
        } else {
            self.expand_userdefined_macro(call)
        }
    }

    fn try_expand_predefined_macro(&self, call: &MacroCall) -> Result<Option<LexicalToken>> {
        let expanded = match call.name().as_str().get() {
            "FILE" => {
                let span = call.span();
                let current = span.start();
                let filename = {
                    self.codemap
                        .lock()
                        .unwrap()
                        .find_file(current)
                        .unwrap()
                        .name()
                        .clone()
                };
                LexicalToken(
                    current,
                    Token::String(Symbol::intern(&filename.to_string())),
                    span.end(),
                )
            }
            "LINE" => {
                let span = call.span();
                let current = span.start();
                let line = {
                    self.codemap
                        .lock()
                        .unwrap()
                        .find_file(current)
                        .unwrap()
                        .find_line(current)
                        .unwrap()
                };
                let line = line.to_usize() as i64;
                LexicalToken(current, Token::Integer(line), span.end())
            }
            "MACHINE" => {
                let span = call.span();
                let current = span.start();
                LexicalToken(current, Token::Atom(Symbol::intern("Lumen")), span.end())
            }
            _ => return Ok(None),
        };
        Ok(Some(expanded))
    }

    fn expand_userdefined_macro(&self, call: MacroCall) -> Result<VecDeque<LexicalToken>> {
        let definition = match self.macros.get(&call.name()) {
            None => return Err(PreprocessorError::UndefinedMacro(call)),
            Some(def) => def,
        };
        match *definition {
            MacroDef::Dynamic(ref replacement) => Ok(replacement.clone().into()),
            MacroDef::String(ref s) => Ok(vec![LexicalToken(
                ByteIndex(0),
                Token::String(s.clone()),
                ByteIndex(0),
            )]
            .into()),
            MacroDef::Boolean(true) => Ok(vec![LexicalToken(
                ByteIndex(0),
                Token::Atom(symbols::True),
                ByteIndex(0),
            )]
            .into()),
            MacroDef::Boolean(false) => Ok(VecDeque::new()),
            MacroDef::Static(ref def) => {
                let arity = def.variables.as_ref().map(|v| v.len()).unwrap_or(0);
                let argc = call.args.as_ref().map(|a| a.len()).unwrap_or(0);
                if arity != argc {
                    let err = format!(
                        "expected {} arguments at call site, but given {}",
                        arity, argc
                    );
                    return Err(PreprocessorError::BadMacroCall(
                        call,
                        definition.clone(),
                        err,
                    ));
                }
                let bindings = def
                    .variables
                    .as_ref()
                    .iter()
                    .flat_map(|i| i.iter().map(|v| v.symbol()))
                    .zip(
                        call.args
                            .iter()
                            .flat_map(|i| i.iter().map(|a| &a.tokens[..])),
                    )
                    .collect::<HashMap<_, _>>();
                let expanded = self.expand_replacement(bindings, &def.replacement)?;
                Ok(expanded)
            }
        }
    }

    fn expand_replacement(
        &self,
        bindings: HashMap<Symbol, &[LexicalToken]>,
        replacement: &[LexicalToken],
    ) -> Result<VecDeque<LexicalToken>> {
        let mut expanded = VecDeque::new();
        let replacement_tokens: VecDeque<_> = replacement.iter().map(|t| Ok(t.clone())).collect();
        let mut reader = TokenBufferReader::new(self.codemap.clone(), replacement_tokens);

        loop {
            if let Some(call) = reader.try_read_macro_call(&self.macros)? {
                let nested = self.expand_macro(call)?;
                for token in nested.into_iter().rev() {
                    reader.unread_token(token);
                }
            } else if let Some(stringify) = reader.try_read::<Stringify>()? {
                let tokens = match bindings.get(&stringify.name.symbol()) {
                    None => return Err(PreprocessorError::UndefinedStringifyMacro(stringify)),
                    Some(tokens) => tokens,
                };
                let string = tokens.iter().map(|t| t.to_string()).collect::<String>();
                let span = tokens[0].span();
                let start = span.start();
                let end = span.end();
                let token = (start, Token::String(Symbol::intern(&string)), end);
                expanded.push_back(token.into());
            } else if let Some(token) = reader.try_read_token()? {
                match IdentToken::try_from(token.clone()) {
                    Ok(ident) => match bindings.get(&ident.symbol()) {
                        Some(value) => {
                            let nested = self.expand_replacement(HashMap::new(), value)?;
                            expanded.extend(nested);
                            continue;
                        }
                        None => (),
                    },
                    Err(_) => (),
                }
                expanded.push_back(token);
            } else {
                break;
            }
        }
        Ok(expanded)
    }

    fn try_read_directive(&mut self) -> Result<Option<Directive>> {
        let directive: Directive = if let Some(directive) = self.reader.try_read()? {
            directive
        } else {
            return Ok(None);
        };

        let ignore = self.ignore();
        match directive {
            Directive::Module(ref d) => {
                self.macros
                    .insert(symbols::Module, MacroDef::String(d.name.symbol()));
            }
            Directive::Include(ref d) if !ignore => {
                let path = d.include();
                self.reader.inject_include(path)?;
            }
            Directive::IncludeLib(ref d) if !ignore => {
                let path = d.include_lib(&self.code_paths)?;
                self.reader.inject_include(path)?;
            }
            Directive::Define(ref d) if !ignore => {
                self.macros
                    .insert(d.name.symbol(), MacroDef::Static(d.clone()));
            }
            Directive::Undef(ref d) if !ignore => {
                self.macros.remove(&d.name());
            }
            Directive::Ifdef(ref d) => {
                let entered = self.macros.contains_key(&d.name());
                self.branches.push(Branch::new(entered));
            }
            Directive::If(ref d) => {
                let entered = self.eval_conditional(d.span(), d.condition.clone())?;
                self.branches.push(Branch::new(entered));
            }
            Directive::Ifndef(ref d) => {
                let entered = !self.macros.contains_key(&d.name());
                self.branches.push(Branch::new(entered));
            }
            Directive::Else(_) => match self.branches.last_mut() {
                None => return Err(PreprocessorError::OrphanedElse(directive)),
                Some(branch) => {
                    match branch.switch_to_else_branch() {
                        Err(_) => return Err(PreprocessorError::OrphanedElse(directive)),
                        Ok(_) => (),
                    };
                }
            },
            Directive::Elif(ref d) => {
                // Treat this like -endif followed by -if(Cond)
                match self.branches.pop() {
                    None => return Err(PreprocessorError::OrphanedElse(directive)),
                    Some(_) => {
                        let entered = self.eval_conditional(d.span(), d.condition.clone())?;
                        self.branches.push(Branch::new(entered));
                    }
                }
            }
            Directive::Endif(_) => match self.branches.pop() {
                None => return Err(PreprocessorError::OrphanedEnd(directive)),
                Some(_) => (),
            },
            Directive::Error(ref d) if !ignore => {
                let span = d.span();
                let err = d.message.symbol().as_str().get().to_string();
                return Err(PreprocessorError::CompilerError(Some(span), err));
            }
            Directive::Warning(ref d) if !ignore => {
                if self.no_warn {
                    return Ok(Some(directive));
                }
                if self.warnings_as_errors {
                    let span = d.span();
                    let err = d.message.symbol().as_str().get().to_string();
                    return Err(PreprocessorError::CompilerError(Some(span), err));
                }
                let span = d.span();
                let warn = d.message.symbol().as_str().get();
                let diag = Diagnostic::new_warning("found warning directive")
                    .with_label(Label::new_primary(span).with_message(warn));
                let emitter =
                    StandardStreamEmitter::new(ColorChoice::Auto).set_codemap(self.codemap.clone());
                emitter.diagnostic(&diag).unwrap();
            }
            _ => {}
        }
        Ok(Some(directive))
    }

    fn eval_conditional(&self, span: ByteSpan, condition: VecDeque<Lexed>) -> Result<bool> {
        use crate::lexer::{symbols, Ident};
        use crate::parser::ast::{Expr, Literal};
        use crate::parser::Parse;
        use crate::preprocessor::evaluator;

        let pp = self.clone_with(condition);
        let result = <Expr as Parse<Expr>>::parse_tokens(pp);
        match result {
            Ok(expr) => match evaluator::eval(expr)? {
                Expr::Literal(Literal::Atom(Ident { ref name, .. })) if *name == symbols::True => {
                    Ok(true)
                }
                Expr::Literal(Literal::Atom(Ident { ref name, .. })) if *name == symbols::False => {
                    Ok(false)
                }
                _other => return Err(PreprocessorError::InvalidConditional(span)),
            },
            Err(errs) => return Err(PreprocessorError::ParseError(span, errs)),
        }
    }
}

impl<R, S> Iterator for Preprocessor<R>
where
    R: TokenReader<Source = S>,
{
    type Item = Preprocessed;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Err(e) => Some(Err(e)),
            Ok(None) => None,
            Ok(Some(token)) => Some(Ok(token.into())),
        }
    }
}

#[derive(Debug)]
struct Branch {
    pub then_branch: bool,
    pub entered: bool,
}
impl Branch {
    pub fn new(entered: bool) -> Self {
        Branch {
            then_branch: true,
            entered,
        }
    }
    pub fn switch_to_else_branch(&mut self) -> std::result::Result<(), Error> {
        if !self.then_branch {
            return Err(format_err!("orphaned else"));
        }
        self.then_branch = false;
        self.entered = !self.entered;
        Ok(())
    }
}
