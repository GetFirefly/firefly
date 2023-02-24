use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;

use firefly_intern::{symbols, Symbol};
use firefly_parser::Source;
use firefly_util::diagnostics::*;

use crate::ast::Literal;
use crate::evaluator;
use crate::lexer::Lexer;
use crate::lexer::{DelayedSubstitution, IdentToken, Lexed, LexicalToken, Token};
use crate::parser::{Parser, ParserError};

use super::macros::Stringify;
use super::token_reader::{TokenBufferReader, TokenReader, TokenStreamReader};
use super::{Directive, MacroCall, MacroContainer, MacroDef, MacroIdent};
use super::{Preprocessed, PreprocessorError, Result as PResult};

pub struct Preprocessor<'a, Reader: TokenReader> {
    diagnostics: &'a DiagnosticsHandler,
    codemap: Arc<CodeMap>,
    reader: Reader,
    can_directive_start: bool,
    code_paths: VecDeque<PathBuf>,
    include_paths: VecDeque<PathBuf>,
    branches: Vec<Branch>,
    macros: MacroContainer,
    macro_calls: BTreeMap<SourceIndex, MacroCall>,
    expanded_tokens: VecDeque<LexicalToken>,
    warnings_as_errors: bool,
    no_warn: bool,
}
impl<'a, S> Preprocessor<'a, TokenStreamReader<S>>
where
    S: Source,
{
    pub fn new(parser: &Parser, tokens: Lexer<S>, diagnostics: &'a DiagnosticsHandler) -> Self {
        let reader = TokenStreamReader::new(parser.codemap.clone(), tokens);
        let code_paths = parser.config.code_paths.clone();
        let include_paths = parser.config.include_paths.clone();

        let mut macros = match parser.config.macros {
            None => MacroContainer::new(),
            Some(ref macros) => macros.clone(),
        };
        macros.insert(
            MacroIdent::Const(Symbol::intern("FUNCTION_NAME")),
            MacroDef::DelayedSubstitution(DelayedSubstitution::FunctionName),
        );
        macros.insert(
            MacroIdent::Const(Symbol::intern("FUNCTION_ARITY")),
            MacroDef::DelayedSubstitution(DelayedSubstitution::FunctionArity),
        );
        macros.insert(
            MacroIdent::Func(Symbol::intern("FEATURE_AVAILABLE"), 1),
            MacroDef::Dynamic(vec![]),
        );
        macros.insert(
            MacroIdent::Func(Symbol::intern("FEATURE_ENABLED"), 1),
            MacroDef::Dynamic(vec![]),
        );

        Self {
            diagnostics,
            codemap: parser.codemap.clone(),
            reader,
            can_directive_start: true,
            code_paths,
            include_paths,
            branches: Vec::new(),
            macros,
            macro_calls: BTreeMap::new(),
            expanded_tokens: VecDeque::new(),
            warnings_as_errors: parser.config.warnings_as_errors,
            no_warn: parser.config.no_warn,
        }
    }
}
impl<'a, R, S> Preprocessor<'a, R>
where
    R: TokenReader<Source = S>,
{
    fn clone_with(&self, tokens: VecDeque<Lexed>) -> Preprocessor<'a, TokenBufferReader> {
        let codemap = self.codemap.clone();
        let reader = TokenBufferReader::new(codemap.clone(), tokens);
        Preprocessor {
            diagnostics: self.diagnostics,
            codemap,
            reader,
            can_directive_start: false,
            code_paths: self.code_paths.clone(),
            include_paths: self.include_paths.clone(),
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

    fn next_token(&mut self) -> Result<Option<LexicalToken>, ParserError> {
        loop {
            if let Some(token) = self.expanded_tokens.pop_front() {
                return Ok(Some(token));
            }
            if self.can_directive_start {
                match self.reader.try_read().map_err(ParserError::from)? {
                    Some(directive) => {
                        self.try_handle_directive(directive)
                            .map_err(ParserError::from)?;
                        continue;
                    }
                    None => (),
                }
            }
            if !self.ignore() {
                if let Some(m) = self
                    .reader
                    .try_read_macro_call(&self.macros)
                    .map_err(ParserError::from)?
                {
                    self.macro_calls.insert(m.span().start(), m.clone());
                    self.expanded_tokens = self.expand_macro(m).map_err(ParserError::from)?;
                    continue;
                }
            }
            if let Some(token) = self.reader.try_read_token().map_err(ParserError::from)? {
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

    fn expand_macro(&mut self, call: MacroCall) -> PResult<VecDeque<LexicalToken>> {
        if let Some(expanded) = self.try_expand_predefined_macro(&call)? {
            Ok(vec![expanded].into())
        } else {
            self.expand_userdefined_macro(call)
        }
    }

    fn try_expand_predefined_macro(&mut self, call: &MacroCall) -> PResult<Option<LexicalToken>> {
        let expanded = match call.name().as_str().get() {
            "FILE" => {
                let span = call.span();
                let source_id = span.source_id();
                let current = span.start();
                let file = self.codemap.get(source_id).unwrap();
                let filename = file.name().to_string();
                LexicalToken(
                    current,
                    Token::String(Symbol::intern(&filename)),
                    span.end(),
                )
            }
            "LINE" => {
                let span = call.span();
                let source_id = span.source_id();
                let current = span.start();
                let file = self.codemap.get(source_id).unwrap();
                let line = file.line_index(current.index()).to_usize() as i64;
                LexicalToken(current, Token::Integer(line.into()), span.end())
            }
            "MACHINE" => {
                let span = call.span();
                let current = span.start();
                LexicalToken(current, Token::Atom(Symbol::intern("Firefly")), span.end())
            }
            "OTP_RELEASE" => {
                let span = call.span();
                let current = span.start();
                LexicalToken(
                    current,
                    Token::Integer(crate::otp_release().into()),
                    span.end(),
                )
            }
            "FEATURE_AVAILABLE" => {
                let span = call.span();
                match call.args.as_ref() {
                    Some(args) if args.len() == 1 => {
                        let mut iter = args.iter();
                        let arg = iter.next().unwrap();
                        match arg.tokens.as_slice() {
                            [LexicalToken(_, Token::Atom(feature), _)] => {
                                match crate::features::get(feature) {
                                    Some(_) => LexicalToken(
                                        span.start(),
                                        Token::Atom(symbols::True),
                                        span.end(),
                                    ),
                                    None => LexicalToken(
                                        span.start(),
                                        Token::Atom(symbols::False),
                                        span.end(),
                                    ),
                                }
                            }
                            _ => {
                                self.diagnostics.diagnostic(Severity::Warning)
                                    .with_message("invalid call to ?FEATURE_AVAILABLE")
                                    .with_primary_label(span, "expected feature name to be an atom, this feature will be considered unavailable")
                                    .emit();
                                LexicalToken(span.start(), Token::Atom(symbols::False), span.end())
                            }
                        }
                    }
                    None | Some(_) => {
                        self.diagnostics.diagnostic(Severity::Warning)
                            .with_message("invalid call to ?FEATURE_AVAILABLE")
                            .with_primary_label(span, "this macro requires a single feature name as its argument, this feature will be considered unavailable")
                            .emit();
                        LexicalToken(span.start(), Token::Atom(symbols::False), span.end())
                    }
                }
            }
            "FEATURE_ENABLED" => {
                let span = call.span();
                match call.args.as_ref() {
                    Some(args) if args.len() == 1 => {
                        let mut iter = args.iter();
                        let arg = iter.next().unwrap();
                        match arg.tokens.as_slice() {
                            [LexicalToken(_, Token::Atom(feature), _)] => {
                                match crate::features::get(feature) {
                                    Some(feat) if feat.enabled => LexicalToken(
                                        span.start(),
                                        Token::Atom(symbols::True),
                                        span.end(),
                                    ),
                                    Some(_) => LexicalToken(
                                        span.start(),
                                        Token::Atom(symbols::False),
                                        span.end(),
                                    ),
                                    _ => {
                                        let msg = format!("unrecognized feature {}", &feature);
                                        self.diagnostics.diagnostic(Severity::Warning)
                                            .with_message(msg)
                                            .with_primary_label(span, "this is not a recognized feature, it may be unimplemented, or may be a typo, defaulting to disabled")
                                            .emit();
                                        LexicalToken(
                                            span.start(),
                                            Token::Atom(symbols::False),
                                            span.end(),
                                        )
                                    }
                                }
                            }
                            _ => {
                                self.diagnostics.diagnostic(Severity::Warning)
                                    .with_message("invalid call to ?FEATURE_ENABLED")
                                    .with_primary_label(span, "expected feature name to be an atom, this feature will be considered disabled")
                                    .emit();
                                LexicalToken(span.start(), Token::Atom(symbols::False), span.end())
                            }
                        }
                    }
                    None | Some(_) => {
                        self.diagnostics.diagnostic(Severity::Warning)
                            .with_message("invalid call to ?FEATURE_ENABLED")
                            .with_primary_label(span, "this macro requires a single feature name as its argument, this feature will be considered disabled")
                            .emit();
                        LexicalToken(span.start(), Token::Atom(symbols::False), span.end())
                    }
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(expanded))
    }

    fn expand_userdefined_macro(&mut self, call: MacroCall) -> PResult<VecDeque<LexicalToken>> {
        let span = call.span();
        let definition = match self.macros.get(&call) {
            None => return Err(PreprocessorError::UndefinedMacro { call }),
            Some(def) => def.clone(),
        };
        match definition {
            MacroDef::Dynamic(replacement) => {
                let mut replacement = replacement.clone();
                for token in replacement.iter_mut() {
                    token.0 = span.start();
                    token.2 = span.end();
                }
                Ok(replacement.into())
            }
            MacroDef::Atom(s) => Ok(vec![LexicalToken(
                span.start(),
                Token::Atom(s.clone()),
                span.end(),
            )]
            .into()),
            MacroDef::String(s) => Ok(vec![LexicalToken(
                span.start(),
                Token::String(s.clone()),
                span.end(),
            )]
            .into()),
            MacroDef::Boolean(true) => Ok(vec![LexicalToken(
                span.start(),
                Token::Atom(symbols::True),
                span.end(),
            )]
            .into()),
            MacroDef::Boolean(false) => Ok(VecDeque::new()),
            MacroDef::Static(def) => {
                let arity = def.variables.as_ref().map(|v| v.len()).unwrap_or(0);
                let argc = call.args.as_ref().map(|a| a.len()).unwrap_or(0);
                if arity != argc {
                    let err = format!(
                        "expected {} arguments at call site, but given {}",
                        arity, argc
                    );
                    return Err(PreprocessorError::BadMacroCall {
                        call,
                        def: MacroDef::Static(def),
                        reason: err,
                    });
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
                let mut expanded = self.expand_replacement(bindings, &def.replacement)?;
                for token in expanded.iter_mut() {
                    token.0 = span.start();
                    token.2 = span.end();
                }
                Ok(expanded)
            }
            MacroDef::DelayedSubstitution(subst) => Ok(vec![LexicalToken(
                span.start(),
                Token::DelayedSubstitution(subst),
                span.end(),
            )]
            .into()),
        }
    }

    fn expand_replacement(
        &mut self,
        bindings: HashMap<Symbol, &[LexicalToken]>,
        replacement: &[LexicalToken],
    ) -> PResult<VecDeque<LexicalToken>> {
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
                    None => {
                        return Err(PreprocessorError::UndefinedStringifyMacro { call: stringify })
                    }
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

    fn try_handle_directive(&mut self, directive: Directive) -> Result<(), PreprocessorError> {
        let ignore = self.ignore();
        match directive {
            Directive::Module(ref d) => {
                self.macros.insert(
                    MacroIdent::Const(symbols::MODULE),
                    MacroDef::Atom(d.name.symbol()),
                );
                self.macros.insert(
                    MacroIdent::Const(symbols::MODULE_STRING),
                    MacroDef::String(d.name.symbol()),
                );
                // We need to expand this directive back to a token stream for the parser
                self.expanded_tokens = d.expand();
            }
            Directive::Include(ref d) if !ignore => {
                let path = d.include(&self.include_paths)?;
                self.reader.inject_include(path, d.span())?;
            }
            Directive::IncludeLib(ref d) if !ignore => {
                let path = d.include_lib(&self.include_paths, &self.code_paths)?;
                self.reader.inject_include(path, d.span())?;
            }
            Directive::Define(ref d) if !ignore => {
                self.macros.insert(d, MacroDef::Static(d.clone()));
            }
            Directive::Undef(ref d) if !ignore => {
                self.macros.undef(&d.name());
            }
            Directive::Ifdef(ref d) => {
                let entered = self.macros.defined(&d.name());
                self.branches.push(Branch::new(entered));
            }
            Directive::If(ref d) => {
                let entered = self.eval_conditional(d.span(), d.condition.clone())?;
                self.branches.push(Branch::new(entered));
            }
            Directive::Ifndef(ref d) => {
                let entered = !self.macros.defined(&d.name());
                self.branches.push(Branch::new(entered));
            }
            Directive::Else(_) => match self.branches.last_mut() {
                None => {
                    return Err(PreprocessorError::OrphanedElse { directive });
                }
                Some(branch) => {
                    match branch.switch_to_else_branch() {
                        Err(_) => {
                            return Err(PreprocessorError::OrphanedElse { directive });
                        }
                        Ok(_) => (),
                    };
                }
            },
            Directive::Elif(ref d) => {
                // Treat this like -endif followed by -if(Cond)
                match self.branches.pop() {
                    None => {
                        return Err(PreprocessorError::OrphanedElse { directive });
                    }
                    Some(_) => {
                        let entered = self.eval_conditional(d.span(), d.condition.clone())?;
                        self.branches.push(Branch::new(entered));
                    }
                }
            }
            Directive::Endif(_) => match self.branches.pop() {
                None => {
                    return Err(PreprocessorError::OrphanedEnd { directive });
                }
                Some(_) => (),
            },
            Directive::Error(ref d) if !ignore => {
                let span = d.span();
                let err = d.message.symbol().as_str().get().to_string();
                return Err(PreprocessorError::CompilerError {
                    span: Some(span),
                    reason: err,
                });
            }
            Directive::Warning(ref d) if !ignore => {
                if self.no_warn {
                    return Ok(());
                }

                if self.warnings_as_errors {
                    return Err(PreprocessorError::WarningDirective {
                        span: d.span(),
                        message: d.message.symbol(),
                        as_error: true,
                    });
                } else {
                    self.diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message(d.message.symbol())
                        .with_primary_span(d.span())
                        .emit();
                }
            }
            Directive::File(ref f) if !ignore => {
                // TODO
                let span = f.span();
                self.diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("-file directive ignored")
                    .with_primary_label(
                        span,
                        "support for the -file directive has not been implemented yet",
                    )
                    .emit();
            }
            _ => {}
        }
        Ok(())
    }

    fn eval_conditional(
        &mut self,
        span: SourceSpan,
        condition: VecDeque<Lexed>,
    ) -> Result<bool, PreprocessorError> {
        use crate::ast::Expr;
        use crate::parser::Parse;

        let result = {
            let pp = self.clone_with(condition);
            Expr::parse_tokens(self.diagnostics, self.codemap.clone(), pp).map_err(|e| {
                PreprocessorError::ParseError {
                    span,
                    inner: Box::new(e),
                }
            })?
        };
        match evaluator::eval_expr(&result, None) {
            Ok(Literal::Atom(atom)) if atom == symbols::True => Ok(true),
            Ok(Literal::Atom(atom)) if atom == symbols::False => Ok(false),
            Err(err) => Err(err.into()),
            _other => Err(PreprocessorError::InvalidConditional { span }),
        }
    }
}

impl<'a, R, S> Iterator for Preprocessor<'a, R>
where
    R: TokenReader<Source = S>,
{
    type Item = Preprocessed;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Err(err) => Some(Err(err)),
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
    pub fn switch_to_else_branch(&mut self) -> Result<(), ()> {
        if !self.then_branch {
            return Err(());
        }
        self.then_branch = false;
        self.entered = !self.entered;
        Ok(())
    }
}
