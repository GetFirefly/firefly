use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail};
use either::Either;
use either::Either::*;

use firefly_beam::serialization::etf;
use firefly_binary::BitVec;
use firefly_intern::{symbols, Ident};
use firefly_pass::Pass;
use firefly_syntax_base::{DeprecatedFlag, Deprecation, FunctionName};
use firefly_syntax_pp::ast as abstr;
use firefly_syntax_pp::ast::Node;
use firefly_util::diagnostics::*;

use crate::ast::*;

pub struct AbstractErlangToAst<'p> {
    diagnostics: &'p DiagnosticsHandler,
    codemap: &'p CodeMap,
}
impl<'p> AbstractErlangToAst<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler, codemap: &'p CodeMap) -> Self {
        Self {
            diagnostics,
            codemap,
        }
    }
}
impl<'p> Pass for AbstractErlangToAst<'p> {
    type Input<'a> = abstr::Ast;
    type Output<'a> = Module;

    fn run<'a>(&mut self, mut ast: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // The forms we receive should contain an initial file declaration followed by the module declaration,
        let mut source_id = match ast.forms.get(0) {
            Some(abstr::Form::File(ref file)) => self.file_to_source_id(file)?,
            None => bail!("expected at least two forms: file and module declarations"),
            _ => bail!("expected file declaration to be the first form"),
        };

        let module_name = match ast.forms.get(1) {
            Some(abstr::Form::Module(attr)) => {
                let span = self.loc_to_span(source_id, attr.loc());
                Ident::new(attr.name, span)
            }
            None => bail!("expected at least two forms: file and module declarations"),
            _ => bail!("expected module declaration to be the second form"),
        };

        let mut tls = Vec::with_capacity(ast.forms.len() - 2);
        for form in ast.forms.drain(..).skip(2) {
            match self.translate_form(source_id, form, &mut tls) {
                None => continue,
                Some(Err(err)) => return Err(err),
                Some(Ok(Left(tl))) => {
                    tls.push(tl);
                }
                Some(Ok(Right(new_source_id))) => {
                    source_id = new_source_id;
                }
            }
        }

        Ok(Module::new_with_forms(
            &self.diagnostics,
            module_name.span(),
            module_name,
            tls,
        ))
    }
}

impl<'p> AbstractErlangToAst<'p> {
    fn translate_form(
        &mut self,
        source_id: SourceId,
        form: abstr::Form,
        top_levels: &mut [TopLevel],
    ) -> Option<anyhow::Result<Either<TopLevel, SourceId>>> {
        match form {
            // Duplicate module declaration
            abstr::Form::Module(attr) => Some(Err(anyhow!(
                "conflicting module declaration found for: {}",
                attr.name
            ))),
            // We can ignore the eof marker as it is always the last form
            abstr::Form::Eof(_) => None,
            // Report warnings when encountered
            abstr::Form::Warning(warn) => {
                let span = self.loc_to_span(source_id, warn.loc());
                match warn.message {
                    etf::Term::String(s) => {
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message(s.value.as_str().get())
                            .with_primary_span(span)
                            .emit();
                    }
                    etf::Term::Atom(s) => {
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message(s.name.as_str().get())
                            .with_primary_span(span)
                            .emit();
                    }
                    etf::Term::List(charlist) => {
                        let chars = charlist
                            .elements
                            .iter()
                            .map(|t| match t {
                                etf::Term::Integer(i) => i.to_char().ok_or(()),
                                _ => Err(()),
                            })
                            .try_collect::<String>();
                        let message = match chars {
                            Err(_) => "compiler warning".to_string(),
                            Ok(message) => message,
                        };
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message(message)
                            .with_primary_span(span)
                            .emit();
                    }
                    term => {
                        let message = format!("{}", &term);
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message(message)
                            .with_primary_span(span)
                            .emit();
                    }
                }
                None
            }
            // We've encountered a new -file declaration, so load the source for subsequent forms
            abstr::Form::File(ref file) => match self.file_to_source_id(file) {
                Ok(new_source_id) => Some(Ok(Right(new_source_id))),
                Err(err) => Some(Err(err)),
            },
            abstr::Form::Behaviour(attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let name = Ident::new(attr.name, span);
                Some(Ok(Left(TopLevel::Attribute(Attribute::Behaviour(
                    span, name,
                )))))
            }
            abstr::Form::Callback(attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                Some(Ok(Left(TopLevel::Attribute(Attribute::Callback(
                    Callback {
                        span,
                        optional: false,
                        module: attr.name.module.map(|m| Ident::new(m, span)),
                        function: Ident::new(attr.name.name, span),
                        sigs: self.abstr_spec_clause_to_type_sigs(source_id, attr.clauses),
                    },
                )))))
            }
            abstr::Form::OptionalCallbacks(mut attr) => {
                // Find any referenced callbacks already seen and mark them optional
                for fun in attr.funs.drain(..).map(|name| FunctionName {
                    module: name.module,
                    function: name.name,
                    arity: name.arity,
                }) {
                    for tl in top_levels.iter_mut() {
                        match tl {
                            TopLevel::Attribute(Attribute::Callback(ref mut cb)) => {
                                if cb.is_impl(&fun) {
                                    cb.optional = true;
                                    // Once we've seen a callback, we don't need to check remaining forms
                                    break;
                                }
                            }
                            _ => continue,
                        }
                    }
                }
                None
            }
            abstr::Form::Spec(attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                Some(Ok(Left(TopLevel::Attribute(Attribute::Spec(TypeSpec {
                    span,
                    module: attr.name.module.map(|m| Ident::new(m, span)),
                    function: Ident::new(attr.name.name, span),
                    sigs: self.abstr_spec_clause_to_type_sigs(source_id, attr.clauses),
                })))))
            }
            abstr::Form::Export(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let funs = attr
                    .funs
                    .drain(..)
                    .map(|f| {
                        Span::new(
                            span,
                            FunctionName {
                                module: None,
                                function: f.name,
                                arity: f.arity,
                            },
                        )
                    })
                    .collect();
                Some(Ok(Left(TopLevel::Attribute(Attribute::Export(span, funs)))))
            }
            abstr::Form::Import(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let module = Ident::new(attr.module, span);
                let funs = attr
                    .funs
                    .drain(..)
                    .map(|f| {
                        Span::new(
                            span,
                            FunctionName {
                                module: Some(module.name),
                                function: f.name,
                                arity: f.arity,
                            },
                        )
                    })
                    .collect();
                Some(Ok(Left(TopLevel::Attribute(Attribute::Import(
                    span, module, funs,
                )))))
            }
            abstr::Form::ExportType(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let types = attr
                    .types
                    .drain(..)
                    .map(|f| {
                        Span::new(
                            span,
                            FunctionName {
                                module: None,
                                function: f.name,
                                arity: f.arity,
                            },
                        )
                    })
                    .collect();
                Some(Ok(Left(TopLevel::Attribute(Attribute::ExportType(
                    span, types,
                )))))
            }
            abstr::Form::Compile(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let expr = if attr.options.len() == 1 {
                    let term = attr.options.pop().unwrap();
                    Expr::Literal(self.etf_term_to_literal(span, term))
                } else {
                    Expr::Literal(
                        attr.options
                            .drain(..)
                            .rfold(Literal::Nil(span), |tail, term| {
                                let head = self.etf_term_to_literal(span, term);
                                Literal::Cons(span, Box::new(head), Box::new(tail))
                            }),
                    )
                };
                Some(Ok(Left(TopLevel::Attribute(Attribute::Compile(
                    span, expr,
                )))))
            }
            abstr::Form::Type(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());

                Some(Ok(Left(TopLevel::Attribute(Attribute::Type(TypeDef {
                    span,
                    opaque: attr.is_opaque,
                    name: Ident::new(attr.name, span),
                    params: attr
                        .vars
                        .drain(..)
                        .map(|v| {
                            Name::Var(Ident::new(v.name, self.loc_to_span(source_id, v.loc())))
                        })
                        .collect(),
                    ty: self.abstr_type_to_type(source_id, &attr.ty),
                })))))
            }
            abstr::Form::OnLoad(attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                Some(Ok(Left(TopLevel::Attribute(Attribute::OnLoad(
                    span,
                    Span::new(span, FunctionName::new_local(attr.fun.name, attr.fun.arity)),
                )))))
            }
            abstr::Form::Nifs(mut attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                let funs = attr
                    .funs
                    .drain(..)
                    .map(|f| Span::new(span, FunctionName::new_local(f.name, f.arity)))
                    .collect();
                Some(Ok(Left(TopLevel::Attribute(Attribute::Nifs(span, funs)))))
            }
            abstr::Form::Attr(attr) => {
                let span = self.loc_to_span(source_id, attr.loc());
                match attr.name {
                    symbols::Deprecated => {
                        // We handle the deprecated attribute here rather than in syntax_pp for simplicity
                        match attr.value {
                            // -deprecated(module).
                            etf::Term::Atom(a) if a.name == symbols::Module => {
                                let deprecations = vec![Deprecation::Module {
                                    span,
                                    flag: DeprecatedFlag::Eventually,
                                }];

                                Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                    deprecations,
                                )))))
                            }
                            etf::Term::Tuple(etf::Tuple { elements }) => {
                                match elements.as_slice() {
                                    // -deprecated({'_', '_'}).
                                    [etf::Term::Atom(f), etf::Term::Atom(a)]
                                        if f.name == symbols::Underscore
                                            && a.name == symbols::Underscore =>
                                    {
                                        let deprecation = Deprecation::Module {
                                            span,
                                            flag: DeprecatedFlag::Eventually,
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // -deprecated({'_', '_', DeprecatedFlag}).
                                    [etf::Term::Atom(f), etf::Term::Atom(a), etf::Term::Atom(flag)]
                                        if f.name == symbols::Underscore
                                            && a.name == symbols::Underscore =>
                                    {
                                        let flag = Ident::new(flag.name, span);
                                        let deprecation = Deprecation::Module {
                                            span,
                                            flag: flag.into(),
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // -deprecated({f, '_'}).
                                    [etf::Term::Atom(f), etf::Term::Atom(a)]
                                        if a.name == symbols::Underscore =>
                                    {
                                        let deprecation = Deprecation::FunctionAnyArity {
                                            span,
                                            name: f.name,
                                            flag: DeprecatedFlag::Eventually,
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // -deprecated({f, Arity}).
                                    [etf::Term::Atom(f), etf::Term::Integer(a)] => {
                                        let function = Span::new(
                                            span,
                                            FunctionName::new_local(f.name, a.to_arity()),
                                        );
                                        let deprecation = Deprecation::Function {
                                            span,
                                            function,
                                            flag: DeprecatedFlag::Eventually,
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // -deprecated({f, Arity, "some message"}).
                                    [etf::Term::Atom(f), etf::Term::Integer(a), etf::Term::String(s)] =>
                                    {
                                        let function = Span::new(
                                            span,
                                            FunctionName::new_local(f.name, a.to_arity()),
                                        );
                                        let deprecation = Deprecation::Function {
                                            span,
                                            function,
                                            flag: DeprecatedFlag::Description(Ident::new(
                                                s.value, span,
                                            )),
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // -deprecated({f, Arity, DeprecatedFlag}).
                                    [etf::Term::Atom(f), etf::Term::Integer(a), etf::Term::Atom(flag)] =>
                                    {
                                        let function = Span::new(
                                            span,
                                            FunctionName::new_local(f.name, a.to_arity()),
                                        );
                                        let flag = Ident::new(flag.name, span);
                                        let deprecation = Deprecation::Function {
                                            span,
                                            function,
                                            flag: flag.into(),
                                        };
                                        Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                            vec![deprecation],
                                        )))))
                                    }
                                    // Ignore unrecognized formats
                                    _ => None,
                                }
                            }
                            etf::Term::List(etf::List { mut elements }) => {
                                let deprecations = elements.drain(..).filter_map(|term| {
                                    match term {
                                        etf::Term::Tuple(etf::Tuple { elements }) => {
                                            // NOTE: We don't accept the module-wide forms here, as that semantically makes no sense
                                            match elements.as_slice() {
                                                // -deprecated([{f, Arity, "some message"}]).
                                                [etf::Term::Atom(f), etf::Term::Integer(a), etf::Term::String(s)] => {
                                                    let function = Span::new(span, FunctionName::new_local(f.name, a.to_arity()));
                                                    let flag = Ident::new(s.value, span);
                                                    Some(Deprecation::Function { span, function, flag: flag.into() })
                                                }
                                                // -deprecated([{f, Arity, DeprecatedFlag}]).
                                                [etf::Term::Atom(f), etf::Term::Integer(a), etf::Term::Atom(s)] => {
                                                    let function = Span::new(span, FunctionName::new_local(f.name, a.to_arity()));
                                                    let flag = Ident::new(s.name, span);
                                                    Some(Deprecation::Function { span, function, flag: flag.into() })
                                                }
                                                _ => None,
                                            }
                                        }
                                        _other => None,
                                    }
                                }).collect::<Vec<_>>();
                                if deprecations.is_empty() {
                                    None
                                } else {
                                    Some(Ok(Left(TopLevel::Attribute(Attribute::Deprecation(
                                        deprecations,
                                    )))))
                                }
                            }
                            // Ignore unrecognized forms
                            _other => None,
                        }
                    }
                    name => {
                        let name = Ident::new(name, span);
                        let value = self.etf_term_to_literal(span, attr.value);
                        Some(Ok(Left(TopLevel::Attribute(Attribute::Custom(
                            UserAttribute {
                                span,
                                name,
                                value: Expr::Literal(value),
                            },
                        )))))
                    }
                }
            }
            abstr::Form::Record(mut def) => {
                let span = self.loc_to_span(source_id, def.loc());
                let name = Ident::new(def.name, span);
                let mut fields = Vec::with_capacity(def.fields.len());
                for field in def.fields.drain(..) {
                    let span = self.loc_to_span(source_id, field.loc());
                    fields.push(RecordField {
                        span,
                        name: Ident::new(field.name, span),
                        value: field
                            .default_value
                            .map(|expr| self.abstr_expr_to_expr(source_id, expr)),
                        ty: Some(self.abstr_type_to_type(source_id, &field.ty)),
                        is_default: field.name == symbols::Underscore,
                    })
                }
                Some(Ok(Left(TopLevel::Record(Record {
                    span,
                    name,
                    fields,
                    default: None,
                }))))
            }
            abstr::Form::Fun(mut def) => {
                let span = self.loc_to_span(source_id, def.loc());
                let name = Ident::new(def.name.name, span);
                let clause_name = Some(Name::Atom(name));
                let clauses = def
                    .clauses
                    .drain(..)
                    .map(|clause| (clause_name, self.abstr_clause_to_clause(source_id, clause)))
                    .collect();
                Some(Ok(Left(TopLevel::Function(Function {
                    span,
                    name,
                    arity: def.name.arity,
                    spec: None,
                    is_nif: false,
                    clauses,
                    var_counter: 0,
                    fun_counter: 0,
                }))))
            }
        }
    }

    fn abstr_clause_to_clause(&mut self, source_id: SourceId, mut clause: abstr::Clause) -> Clause {
        let span = self.loc_to_span(source_id, clause.loc());
        Clause {
            span,
            patterns: clause
                .patterns
                .drain(..)
                .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                .collect(),
            body: clause
                .body
                .drain(..)
                .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                .collect(),
            guards: clause
                .guards
                .drain(..)
                .map(|mut guard| Guard {
                    span,
                    conditions: guard
                        .and_guards
                        .drain(..)
                        .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                        .collect(),
                })
                .collect(),
            compiler_generated: false,
        }
    }

    fn abstr_expr_to_expr(&mut self, source_id: SourceId, expr: abstr::Expression) -> Expr {
        let span = self.loc_to_span(source_id, expr.loc());
        match expr {
            abstr::Expression::Integer(box i) => Expr::Literal(Literal::Integer(span, i.value)),
            abstr::Expression::Float(f) => Expr::Literal(Literal::Float(span, f.value)),
            abstr::Expression::String(box s) => {
                Expr::Literal(Literal::String(Ident::new(s.value, span)))
            }
            abstr::Expression::Char(c) => Expr::Literal(Literal::Char(span, c.value)),
            abstr::Expression::Atom(a) => Expr::Literal(Literal::Atom(Ident::new(a.value, span))),
            abstr::Expression::Match(box expr) => {
                let pattern = self.abstr_expr_to_expr(source_id, expr.left);
                let expr = self.abstr_expr_to_expr(source_id, expr.right);
                Expr::Match(Match {
                    span,
                    pattern: Box::new(pattern),
                    expr: Box::new(expr),
                })
            }
            abstr::Expression::Var(ref v) => Expr::Var(Var(Ident::new(v.name, span))),
            abstr::Expression::Tuple(box mut tuple) => {
                let elements = tuple
                    .elements
                    .drain(..)
                    .map(|e| self.abstr_expr_to_expr(source_id, e))
                    .collect();
                Expr::Tuple(Tuple { span, elements })
            }
            abstr::Expression::Nil(_) => Expr::Literal(Literal::Nil(span)),
            abstr::Expression::Cons(box cons) => {
                let head = self.abstr_expr_to_expr(source_id, cons.head);
                let tail = self.abstr_expr_to_expr(source_id, cons.tail);
                Expr::Cons(Cons {
                    span,
                    head: Box::new(head),
                    tail: Box::new(tail),
                })
            }
            abstr::Expression::Binary(box mut bin) => {
                let span = self.loc_to_span(source_id, bin.loc());
                let elements = bin
                    .elements
                    .drain(..)
                    .map(|element| {
                        let span = self.loc_to_span(source_id, element.loc());
                        let has_size = element.size.is_some();
                        BinaryElement {
                            span,
                            bit_expr: self.abstr_expr_to_expr(source_id, element.element),
                            bit_size: element
                                .size
                                .map(|expr| self.abstr_expr_to_expr(source_id, expr)),
                            specifier: match element.tsl {
                                None => None,
                                Some(mut flags) => {
                                    let bit_types = flags
                                        .drain(..)
                                        .map(|flag| match flag.value {
                                            None => {
                                                BitType::Name(span, Ident::new(flag.name, span))
                                            }
                                            Some(unit) => BitType::Sized(
                                                span,
                                                Ident::new(flag.name, span),
                                                unit.try_into().unwrap(),
                                            ),
                                        })
                                        .collect::<Vec<_>>();

                                    Some(
                                        crate::parser::binary::specifier_from_parsed(
                                            &bit_types, has_size,
                                        )
                                        .unwrap(),
                                    )
                                }
                            },
                        }
                    })
                    .collect();
                Expr::Binary(Binary { span, elements })
            }
            abstr::Expression::UnaryOp(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let op = expr.operator.try_into().unwrap();
                let operand = self.abstr_expr_to_expr(source_id, expr.operand);
                Expr::UnaryExpr(UnaryExpr {
                    span,
                    op,
                    operand: Box::new(operand),
                })
            }
            abstr::Expression::BinaryOp(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let op = expr.operator.try_into().unwrap();
                let lhs = self.abstr_expr_to_expr(source_id, expr.left_operand);
                let rhs = self.abstr_expr_to_expr(source_id, expr.right_operand);
                Expr::BinaryExpr(BinaryExpr {
                    span,
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                })
            }
            abstr::Expression::Record(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let name = Ident::new(expr.name, span);
                let fields = expr
                    .fields
                    .drain(..)
                    .map(|field| {
                        let span = self.loc_to_span(source_id, field.loc());
                        RecordField {
                            span,
                            name: field
                                .name
                                .map(|n| Ident::new(n, span))
                                .unwrap_or_else(|| Ident::new(symbols::Underscore, span)),
                            value: Some(self.abstr_expr_to_expr(source_id, field.value)),
                            ty: None,
                            is_default: field.name.is_none(),
                        }
                    })
                    .collect();
                match expr.base {
                    None => Expr::Record(Record {
                        span,
                        name,
                        fields,
                        default: None,
                    }),
                    Some(base) => Expr::RecordUpdate(RecordUpdate {
                        span,
                        name,
                        record: Box::new(self.abstr_expr_to_expr(source_id, base)),
                        updates: fields,
                    }),
                }
            }
            abstr::Expression::RecordIndex(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let name = Ident::new(expr.record, span);
                let field = Ident::new(expr.field, span);
                Expr::RecordIndex(RecordIndex { span, name, field })
            }
            abstr::Expression::RecordAccess(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let record = self.abstr_expr_to_expr(source_id, expr.base);
                let name = Ident::new(expr.record, span);
                let field = Ident::new(expr.field, span);
                Expr::RecordAccess(RecordAccess {
                    span,
                    record: Box::new(record),
                    name,
                    field,
                })
            }
            abstr::Expression::Map(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let fields = expr
                    .pairs
                    .drain(..)
                    .map(|pair| {
                        let span = self.loc_to_span(source_id, pair.loc());
                        let key = self.abstr_expr_to_expr(source_id, pair.key);
                        let value = self.abstr_expr_to_expr(source_id, pair.value);
                        if pair.is_assoc {
                            MapField::Assoc { span, key, value }
                        } else {
                            MapField::Exact { span, key, value }
                        }
                    })
                    .collect();
                match expr.base {
                    None => Expr::Map(Map { span, fields }),
                    Some(base) => Expr::MapUpdate(MapUpdate {
                        span,
                        map: Box::new(self.abstr_expr_to_expr(source_id, base)),
                        updates: fields,
                    }),
                }
            }
            abstr::Expression::Catch(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let expr = self.abstr_expr_to_expr(source_id, expr.expr);
                Expr::Catch(Catch {
                    span,
                    expr: Box::new(expr),
                })
            }
            abstr::Expression::Call(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let callee = self.abstr_expr_to_expr(source_id, expr.callee);
                let args = expr
                    .args
                    .drain(..)
                    .map(|arg| self.abstr_expr_to_expr(source_id, arg))
                    .collect();
                Expr::Apply(Apply {
                    span,
                    callee: Box::new(callee),
                    args,
                })
            }
            abstr::Expression::Comprehension(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let body = self.abstr_expr_to_expr(source_id, expr.expr);
                let qualifiers = expr
                    .qualifiers
                    .drain(..)
                    .map(|q| match q {
                        abstr::Qualifier::Generator(g) => {
                            let span = self.loc_to_span(source_id, g.loc());
                            Expr::Generator(Generator {
                                span,
                                ty: GeneratorType::Default,
                                pattern: Box::new(self.abstr_expr_to_expr(source_id, g.pattern)),
                                expr: Box::new(self.abstr_expr_to_expr(source_id, g.expr)),
                            })
                        }
                        abstr::Qualifier::BitStringGenerator(g) => {
                            let span = self.loc_to_span(source_id, g.loc());
                            Expr::Generator(Generator {
                                span,
                                ty: GeneratorType::Bitstring,
                                pattern: Box::new(self.abstr_expr_to_expr(source_id, g.pattern)),
                                expr: Box::new(self.abstr_expr_to_expr(source_id, g.expr)),
                            })
                        }
                        abstr::Qualifier::Filter(expr) => self.abstr_expr_to_expr(source_id, expr),
                    })
                    .collect();
                if expr.is_list {
                    Expr::ListComprehension(ListComprehension {
                        span,
                        body: Box::new(body),
                        qualifiers,
                    })
                } else {
                    Expr::BinaryComprehension(BinaryComprehension {
                        span,
                        body: Box::new(body),
                        qualifiers,
                    })
                }
            }
            abstr::Expression::Block(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let body = expr
                    .body
                    .drain(..)
                    .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                    .collect();
                Expr::Begin(Begin { span, body })
            }
            abstr::Expression::If(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let clauses = expr
                    .clauses
                    .drain(..)
                    .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                    .collect();
                Expr::If(If { span, clauses })
            }
            abstr::Expression::Case(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let clauses = expr
                    .clauses
                    .drain(..)
                    .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                    .collect();
                Expr::Case(Case {
                    span,
                    expr: Box::new(self.abstr_expr_to_expr(source_id, expr.expr)),
                    clauses,
                })
            }
            abstr::Expression::Try(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let exprs = expr
                    .body
                    .drain(..)
                    .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                    .collect();
                let clauses = if expr.case_clauses.is_empty() {
                    None
                } else {
                    Some(
                        expr.case_clauses
                            .drain(..)
                            .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                            .collect(),
                    )
                };
                let catch_clauses = if expr.catch_clauses.is_empty() {
                    None
                } else {
                    Some(
                        expr.catch_clauses
                            .drain(..)
                            .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                            .collect(),
                    )
                };
                let after = if expr.after.is_empty() {
                    None
                } else {
                    Some(
                        expr.after
                            .drain(..)
                            .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                            .collect(),
                    )
                };
                Expr::Try(Try {
                    span,
                    exprs,
                    clauses,
                    catch_clauses,
                    after,
                })
            }
            abstr::Expression::Receive(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let clauses = if expr.clauses.is_empty() {
                    None
                } else {
                    Some(
                        expr.clauses
                            .drain(..)
                            .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                            .collect(),
                    )
                };
                let after = match expr.timeout {
                    None => None,
                    Some(timeout) => {
                        let span = self.loc_to_span(source_id, timeout.loc());
                        let timeout = self.abstr_expr_to_expr(source_id, timeout);
                        let body = expr
                            .after
                            .drain(..)
                            .map(|expr| self.abstr_expr_to_expr(source_id, expr))
                            .collect();
                        Some(After {
                            span,
                            timeout: Box::new(timeout),
                            body,
                        })
                    }
                };

                Expr::Receive(Receive {
                    span,
                    clauses,
                    after,
                })
            }
            abstr::Expression::InternalFun(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let name = FunctionName::new_local(expr.function, expr.arity);
                Expr::FunctionVar(FunctionVar::PartiallyResolved(Span::new(span, name)))
            }
            abstr::Expression::ExternalFun(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                match (expr.module, expr.function, expr.arity) {
                    (
                        abstr::Expression::Atom(m),
                        abstr::Expression::Atom(f),
                        abstr::Expression::Integer(a),
                    ) => {
                        let name = FunctionName::new(m.value, f.value, a.value.try_into().unwrap());
                        Expr::FunctionVar(FunctionVar::Resolved(Span::new(span, name)))
                    }
                    (m, f, a) => {
                        let module = match m {
                            abstr::Expression::Atom(m) => Name::Atom(Ident::new(m.value, span)),
                            abstr::Expression::Var(v) => Name::Var(Ident::new(v.name, span)),
                            other => panic!("invalid expression in function capture: {:?}", &other),
                        };
                        let function = match f {
                            abstr::Expression::Atom(f) => Name::Atom(Ident::new(f.value, span)),
                            abstr::Expression::Var(v) => Name::Var(Ident::new(v.name, span)),
                            other => panic!("invalid expression in function capture: {:?}", &other),
                        };
                        let arity = match a {
                            abstr::Expression::Integer(i) => {
                                Arity::Int(i.value.try_into().unwrap())
                            }
                            abstr::Expression::Var(v) => Arity::Var(Ident::new(v.name, span)),
                            other => panic!("invalid expression in function capture: {:?}", &other),
                        };
                        let name = UnresolvedFunctionName {
                            span,
                            module: Some(module),
                            function,
                            arity,
                        };
                        Expr::FunctionVar(FunctionVar::Unresolved(name))
                    }
                }
            }
            abstr::Expression::AnonymousFun(box mut expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let arity = expr.clauses[0].patterns.len().try_into().unwrap();
                match expr.name {
                    None => {
                        let clauses = expr
                            .clauses
                            .drain(..)
                            .map(|clause| self.abstr_clause_to_clause(source_id, clause))
                            .collect();
                        Expr::Fun(Fun::Anonymous(AnonymousFun {
                            span,
                            name: None,
                            arity,
                            clauses,
                        }))
                    }
                    Some(name) => {
                        let self_name = Ident::new(name, span);
                        let clauses = expr
                            .clauses
                            .drain(..)
                            .map(|clause| {
                                (
                                    Name::Var(self_name),
                                    self.abstr_clause_to_clause(source_id, clause),
                                )
                            })
                            .collect();
                        Expr::Fun(Fun::Recursive(RecursiveFun {
                            span,
                            name: None,
                            self_name,
                            arity,
                            clauses,
                        }))
                    }
                }
            }
            abstr::Expression::Remote(box expr) => {
                let span = self.loc_to_span(source_id, expr.loc());
                let module = self.abstr_expr_to_expr(source_id, expr.module);
                let function = self.abstr_expr_to_expr(source_id, expr.function);
                Expr::Remote(Remote {
                    span,
                    module: Box::new(module),
                    function: Box::new(function),
                })
            }
        }
    }

    fn etf_term_to_literal(&mut self, span: SourceSpan, term: etf::Term) -> Literal {
        match term {
            etf::Term::Atom(a) => Literal::Atom(Ident::new(a.name, span)),
            etf::Term::String(s) => Literal::String(Ident::new(s.value, span)),
            etf::Term::Integer(i) => Literal::Integer(span, i),
            etf::Term::Float(f) => Literal::Float(span, f),
            etf::Term::List(mut list) => {
                if list.elements.is_empty() {
                    Literal::Nil(span)
                } else {
                    list.elements
                        .drain(..)
                        .rfold(Literal::Nil(span), |tail, term| {
                            let head = self.etf_term_to_literal(span, term);
                            Literal::Cons(span, Box::new(head), Box::new(tail))
                        })
                }
            }
            etf::Term::ImproperList(mut list) => {
                let last = self.etf_term_to_literal(span, *list.last);
                list.elements.drain(..).rfold(last, |tail, term| {
                    let head = self.etf_term_to_literal(span, term);
                    Literal::Cons(span, Box::new(head), Box::new(tail))
                })
            }
            etf::Term::Tuple(mut tuple) => Literal::Tuple(
                span,
                tuple
                    .elements
                    .drain(..)
                    .map(|t| self.etf_term_to_literal(span, t))
                    .collect(),
            ),
            etf::Term::Map(mut map) => {
                let mut out = BTreeMap::new();
                for (key, value) in map.entries.drain(..) {
                    let key = self.etf_term_to_literal(span, key);
                    let value = self.etf_term_to_literal(span, value);
                    out.insert(key, value);
                }
                Literal::Map(span, out)
            }
            etf::Term::Binary(bin) => Literal::Binary(span, bin.bytes.into()),
            etf::Term::BitBinary(bin) => Literal::Binary(
                span,
                BitVec::from_vec_with_trailing_bits(bin.bytes, bin.tail_bits_size),
            ),
            other => panic!("invalid term value for use in literal context: {}", &other),
        }
    }

    fn abstr_spec_clause_to_type_sigs(
        &mut self,
        source_id: SourceId,
        mut clauses: Vec<abstr::Type>,
    ) -> Vec<TypeSig> {
        let mut sigs = Vec::with_capacity(clauses.len());
        for clause in clauses.drain(..) {
            match clause {
                abstr::Type::AnyFun(ty) => {
                    let span = self.loc_to_span(source_id, ty.loc());
                    sigs.push(TypeSig {
                        span,
                        params: vec![],
                        guards: None,
                        ret: Box::new(
                            ty.return_type
                                .as_ref()
                                .map(|ty| self.abstr_type_to_type(source_id, ty))
                                .unwrap_or_else(|| {
                                    Type::Name(Name::Atom(Ident::new(symbols::Any, span)))
                                }),
                        ),
                    });
                }
                abstr::Type::Function(ty) => {
                    let span = self.loc_to_span(source_id, ty.loc());
                    sigs.push(TypeSig {
                        span,
                        params: ty
                            .args
                            .iter()
                            .map(|ty| self.abstr_type_to_type(source_id, ty))
                            .collect(),
                        guards: if ty.constraints.is_empty() {
                            None
                        } else {
                            Some(
                                ty.constraints
                                    .iter()
                                    .map(|ty| self.abstr_constraint_to_type_guard(source_id, ty))
                                    .collect(),
                            )
                        },
                        ret: Box::new(self.abstr_type_to_type(source_id, &ty.return_type)),
                    });
                }
                invalid => {
                    let span = self.loc_to_span(source_id, invalid.loc());
                    self.diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("invalid callback type")
                        .with_primary_label(span, "expected a function type here")
                        .emit();
                    continue;
                }
            }
        }

        sigs
    }

    fn abstr_type_to_type(&mut self, source_id: SourceId, ty: &abstr::Type) -> Type {
        match ty {
            abstr::Type::Any(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Name(Name::Atom(Ident::new(symbols::Any, span)))
            }
            abstr::Type::Atom(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Name(Name::Atom(Ident::new(ty.value, span)))
            }
            abstr::Type::Integer(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Integer(span, ty.value.clone())
            }
            abstr::Type::Var(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Name(Name::Var(Ident::new(ty.name, span)))
            }
            abstr::Type::Annotated(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let name_span = self.loc_to_span(source_id, ty.name.loc());
                let name = Ident::new(ty.name.name, name_span);
                Type::Annotated {
                    span,
                    name: Name::Atom(name),
                    ty: Box::new(self.abstr_type_to_type(source_id, &ty.ty)),
                }
            }
            abstr::Type::UnaryOp(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let rhs = self.abstr_expression_to_type(source_id, &ty.operand);
                Type::UnaryOp {
                    span,
                    op: ty.operator.try_into().unwrap(),
                    rhs: Box::new(rhs),
                }
            }
            abstr::Type::BinaryOp(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let lhs = self.abstr_expression_to_type(source_id, &ty.left_operand);
                let rhs = self.abstr_expression_to_type(source_id, &ty.right_operand);
                Type::BinaryOp {
                    span,
                    lhs: Box::new(lhs),
                    op: ty.operator.try_into().unwrap(),
                    rhs: Box::new(rhs),
                }
            }
            abstr::Type::BitString(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let size = Type::Integer(span, ty.size.into());
                let tail_bits = Type::Integer(span, ty.tail_bits.into());
                Type::Binary(span, Box::new(size), Box::new(tail_bits))
            }
            abstr::Type::Nil(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Nil(span)
            }
            abstr::Type::AnyFun(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::AnyFun {
                    span,
                    ret: ty
                        .return_type
                        .as_ref()
                        .map(|ty| Box::new(self.abstr_type_to_type(source_id, ty))),
                }
            }
            abstr::Type::Function(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Fun {
                    span,
                    params: ty
                        .args
                        .iter()
                        .map(|ty| self.abstr_type_to_type(source_id, ty))
                        .collect(),
                    ret: Box::new(self.abstr_type_to_type(source_id, &ty.return_type)),
                }
            }
            abstr::Type::Range(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Range {
                    span,
                    start: Box::new(self.abstr_type_to_type(source_id, &ty.low)),
                    end: Box::new(self.abstr_type_to_type(source_id, &ty.high)),
                }
            }
            abstr::Type::Map(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let pairs = ty
                    .pairs
                    .iter()
                    .map(|p| {
                        let span = self.loc_to_span(source_id, p.loc());
                        let key = self.abstr_type_to_type(source_id, &p.key);
                        let value = self.abstr_type_to_type(source_id, &p.value);
                        Type::KeyValuePair(span, Box::new(key), Box::new(value))
                    })
                    .collect();
                Type::Map(span, pairs)
            }
            abstr::Type::BuiltIn(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                if ty.args.is_empty() {
                    Type::Name(Name::Atom(Ident::new(ty.name, span)))
                } else {
                    Type::Generic {
                        span,
                        fun: Ident::new(ty.name, span),
                        params: ty
                            .args
                            .iter()
                            .map(|ty| self.abstr_type_to_type(source_id, ty))
                            .collect(),
                    }
                }
            }
            abstr::Type::Record(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let name = Ident::new(ty.name, span);
                let fields = ty
                    .fields
                    .iter()
                    .map(|ty| {
                        let span = self.loc_to_span(source_id, ty.loc());
                        let name = Ident::new(ty.name, span);
                        Type::Field(
                            span,
                            name,
                            Box::new(self.abstr_type_to_type(source_id, &ty.ty)),
                        )
                    })
                    .collect();
                Type::Record(span, name, fields)
            }
            abstr::Type::Remote(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Remote {
                    span,
                    module: Ident::new(ty.name.module.unwrap(), span),
                    fun: Ident::new(ty.name.name, span),
                    args: ty
                        .args
                        .iter()
                        .map(|ty| self.abstr_type_to_type(source_id, ty))
                        .collect(),
                }
            }
            abstr::Type::AnyTuple(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Name(Name::Atom(Ident::new(symbols::Tuple, span)))
            }
            abstr::Type::Tuple(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                let elements = ty
                    .elements
                    .iter()
                    .map(|ty| self.abstr_type_to_type(source_id, ty))
                    .collect();
                Type::Tuple(span, elements)
            }
            abstr::Type::Union(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                Type::Union {
                    span,
                    types: ty
                        .types
                        .iter()
                        .map(|ty| self.abstr_type_to_type(source_id, ty))
                        .collect(),
                }
            }
            abstr::Type::User(ty) => {
                let span = self.loc_to_span(source_id, ty.loc());
                if ty.args.is_empty() {
                    Type::Name(Name::Atom(Ident::new(ty.name, span)))
                } else {
                    Type::Generic {
                        span,
                        fun: Ident::new(ty.name, span),
                        params: ty
                            .args
                            .iter()
                            .map(|ty| self.abstr_type_to_type(source_id, ty))
                            .collect(),
                    }
                }
            }
            abstr::Type::Product(_) => {
                panic!("product types should not be present in the ast at this point")
            }
        }
    }

    fn abstr_expression_to_type(&mut self, source_id: SourceId, expr: &abstr::Expression) -> Type {
        match expr {
            abstr::Expression::Var(v) => {
                let span = self.loc_to_span(source_id, v.loc());
                Type::Name(Name::Var(Ident::new(v.name, span)))
            }
            abstr::Expression::Atom(a) => {
                let span = self.loc_to_span(source_id, a.loc());
                Type::Name(Name::Atom(Ident::new(a.value, span)))
            }
            abstr::Expression::Integer(i) => {
                let span = self.loc_to_span(source_id, i.loc());
                Type::Integer(span, i.value.clone())
            }
            other => panic!(
                "unsupported expression type in abstract erlang type: {:?}",
                &other
            ),
        }
    }

    fn abstr_constraint_to_type_guard(
        &mut self,
        source_id: SourceId,
        constraint: &abstr::Constraint,
    ) -> TypeGuard {
        let span = self.loc_to_span(source_id, constraint.loc());
        let var_span = self.loc_to_span(source_id, constraint.var.loc());
        let var = Name::Var(Ident::new(constraint.var.name, var_span));
        let ty = self.abstr_type_to_type(source_id, &constraint.subtype);
        TypeGuard { span, var, ty }
    }

    fn file_to_source_id(&mut self, file: &abstr::FileAttr) -> anyhow::Result<SourceId> {
        let path = PathBuf::from(file.original_file.as_str().get());
        let content = std::fs::read_to_string(&path)?;
        Ok(self.codemap.add(path, content))
    }

    #[inline]
    fn loc_to_span(&mut self, source_id: SourceId, loc: abstr::Location) -> SourceSpan {
        self.codemap
            .line_column_to_span(source_id, loc.line - 1, loc.column - 1)
            .unwrap()
    }
}
