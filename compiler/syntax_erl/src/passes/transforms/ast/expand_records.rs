use anyhow::anyhow;

use liblumen_diagnostics::{SourceSpan, Span};
use liblumen_intern::{symbols, Ident};
use liblumen_pass::Pass;
use liblumen_syntax_core as syntax_core;

use crate::ast::*;
use crate::visit::{self, VisitMut};

/// This pass performs expansion of records and record operations into raw
/// tuple operations.
///
/// Once this pass has run, there should no longer be _any_ record expressions
/// in the AST, anywhere. If there are, its an invariant violation and should
/// cause an ICE.
pub struct ExpandRecords<'m> {
    module: &'m Module,
}
impl<'m> ExpandRecords<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self { module }
    }
}
impl<'m> Pass for ExpandRecords<'m> {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut visitor = ExpandRecordsVisitor::new(self.module, f);
        visitor.visit_mut_function(f)?;
        f.var_counter = visitor.var_counter;
        Ok(f)
    }
}

struct ExpandRecordsVisitor<'m> {
    module: &'m Module,
    name: Ident,
    var_counter: usize,
    arity: u8,
    in_pattern: bool,
    in_guard: bool,
    expand_record_info: bool,
}
impl<'m> ExpandRecordsVisitor<'m> {
    fn new(module: &'m Module, f: &Function) -> Self {
        let record_info = syntax_core::FunctionName::new_local(symbols::RecordInfo, 2);
        let expand_record_info = module.functions.get(&record_info).is_none();
        Self {
            module,
            in_pattern: false,
            in_guard: false,
            expand_record_info,
            var_counter: f.var_counter,
            name: f.name,
            arity: f.arity,
        }
    }

    fn next_var(&mut self, span: Option<SourceSpan>) -> Ident {
        let id = self.var_counter;
        self.var_counter += 1;
        let var = format!("${}", id);
        let mut ident = Ident::from_str(&var);
        match span {
            None => ident,
            Some(span) => {
                ident.span = span;
                ident
            }
        }
    }
}
impl<'m> VisitMut for ExpandRecordsVisitor<'m> {
    fn visit_mut_pattern(&mut self, pattern: &mut Expr) -> anyhow::Result<()> {
        self.in_pattern = true;
        visit::visit_mut_pattern(self, pattern)?;
        self.in_pattern = false;
        Ok(())
    }

    fn visit_mut_clause(&mut self, clause: &mut Clause) -> anyhow::Result<()> {
        // TODO: Once the clause has been visited, convert any is_record calls
        // to pattern matches if possible, as these can be better optimized away
        // later to avoid redundant checks
        visit::visit_mut_clause(self, clause)
    }

    fn visit_mut_guard(&mut self, guard: &mut Guard) -> anyhow::Result<()> {
        self.in_guard = true;
        visit::visit_mut_guard(self, guard)?;
        self.in_guard = false;
        Ok(())
    }

    fn visit_mut_expr(&mut self, expr: &mut Expr) -> anyhow::Result<()> {
        match expr {
            // Expand calls to record_info/2 if not shadowed
            Expr::Apply(ref mut apply) if self.expand_record_info => {
                visit::visit_mut_apply(self, apply)?;
                if let Some(callee) = apply.callee.as_ref().as_atom() {
                    if callee.name != symbols::RecordInfo || apply.args.len() != 2 {
                        return Ok(());
                    }

                    let prop = &apply.args[0];
                    let record_name = &apply.args[1];
                    if let Ok(info) = self.try_expand_record_info(record_name, prop) {
                        *expr = info;
                    }
                }
                Ok(())
            }
            // Record creation, or pattern match
            Expr::Record(ref mut record) => {
                // Work inside-out, so visit the record body first
                visit::visit_mut_record(self, record)?;
                // Convert this record into a tuple expression
                let tuple = self.expand_record(record)?;
                // Replace the original expression
                *expr = tuple;
                Ok(())
            }
            // Accessing a record field value, e.g. Expr#myrec.field1
            Expr::RecordAccess(ref mut access) => {
                // Convert this to:
                //
                // case Expr of
                //   {myrec, .., _Field1, ...} ->
                //     _Field1;
                //   _0 ->
                //     erlang:error({badrecord, _0})
                // end
                visit::visit_mut_record_access(self, access)?;
                let expanded = self.expand_access(access)?;
                *expr = expanded;
                Ok(())
            }
            // Referencing a record fields index, e.g. #myrec.field1
            Expr::RecordIndex(ref record_index) => {
                // Convert this to a literal
                let literal = self.expand_index(record_index)?;
                *expr = literal;
                Ok(())
            }
            // Update a record field value, e.g. Expr#myrec{field1=ValueExpr}
            Expr::RecordUpdate(ref mut update) => {
                assert!(!self.in_guard, "record updates are not valid in guards");
                assert!(!self.in_pattern, "record updates are not valid in patterns");
                // Convert this to:
                //
                // case Expr of
                //   {myrec, _, ..} ->
                //     erlang:setelement(N, erlang:setelement(N - 1, .., ..), $ValueN);
                //   $0 ->
                //     erlang:error({badrecord, $0})
                // end
                visit::visit_mut_record_update(self, update)?;
                let expanded = self.expand_update(update)?;
                *expr = expanded;
                Ok(())
            }
            _ => visit::visit_mut_expr(self, expr),
        }
    }
}
impl<'m> ExpandRecordsVisitor<'m> {
    fn try_expand_record_info(&self, record_name: &Expr, prop: &Expr) -> anyhow::Result<Expr> {
        let record_name = record_name.as_atom().ok_or_else(|| {
            anyhow!(
                "expected atom name in call to record_info/2, got '{:?}'",
                record_name
            )
        })?;
        let prop = prop.as_atom().ok_or_else(|| {
            anyhow!(
                "expected the atom 'size' or 'fields' in call to record_info/2, got '{:?}'",
                record_name
            )
        })?;
        let span = record_name.span;

        if let Some(definition) = self.module.record(record_name.name) {
            let prop_name = prop.as_str().get();
            match prop_name {
                "size" => Ok(Expr::Literal(Literal::Integer(
                    span,
                    (1 + definition.fields.len()).into(),
                ))),
                "fields" => {
                    let field_name_list = definition.fields.iter().rev().fold(
                        Expr::Literal(Literal::Nil(span)),
                        |tail, f| {
                            let field_name = Expr::Literal(Literal::Atom(f.name));
                            Expr::Cons(Cons {
                                span,
                                head: Box::new(field_name),
                                tail: Box::new(tail),
                            })
                        },
                    );
                    Ok(field_name_list)
                }
                _ => Err(anyhow!(
                    "expected the atom 'size' or 'fields' in call to record_info/2, but got '{}'",
                    &prop_name
                )),
            }
        } else {
            Err(anyhow!(
                "unable to expand record info for '{}', no such record",
                record_name
            ))
        }
    }

    fn expand_record(&self, record: &Record) -> anyhow::Result<Expr> {
        let name = record.name;
        let symbol = name.name;
        let definition = self
            .module
            .record(symbol)
            .ok_or_else(|| anyhow!("use of undefined record '{}'", name))?;
        let span = record.span.clone();
        let mut elements = Vec::with_capacity(definition.fields.len());
        // The first element is the record name atom
        elements.push(Expr::Literal(Literal::Atom(Ident::with_empty_span(symbol))));
        // For each definition, in order the fields are declared, fetch the corresponding value
        // from the record expression, then depending on whether this is a pattern match or a
        // constructor, we construct an appropriate tuple element expression
        for defined in definition.fields.iter() {
            let provided = record.fields.iter().find_map(|f| {
                if f.name == defined.name {
                    f.value.as_ref()
                } else {
                    None
                }
            });
            if let Some(value_expr) = provided {
                elements.push(value_expr.clone());
            } else if self.in_pattern {
                // This is a pattern, so elided fields need a wildcard pattern
                elements.push(Expr::Var(
                    Ident::with_empty_span(symbols::Underscore).into(),
                ));
            } else {
                // This is a constructor, so use the default initializer, or the atom 'undefined' if one isn't present
                if let Some(default_expr) = defined.value.as_ref() {
                    elements.push(default_expr.clone());
                } else {
                    elements.push(Expr::Literal(Literal::Atom(Ident::with_empty_span(
                        symbols::Undefined,
                    ))));
                }
            }
        }

        Ok(Expr::Tuple(Tuple { span, elements }))
    }

    fn expand_index(&self, record_index: &RecordIndex) -> anyhow::Result<Expr> {
        let name = record_index.name;
        let field = record_index.field;

        let definition = self
            .module
            .record(name.name)
            .ok_or_else(|| anyhow!("reference to undefined record '{}'", name))?;
        let index = definition
            .fields
            .iter()
            .position(|f| f.name == field)
            .ok_or_else(|| {
                anyhow!(
                    "reference to undefined field '{}' of record '{}'",
                    field,
                    name
                )
            })?;
        Ok(Expr::Literal(Literal::Integer(
            record_index.span.clone(),
            (index + 1).into(),
        )))
    }

    fn expand_access(&mut self, record_access: &RecordAccess) -> anyhow::Result<Expr> {
        let name = record_access.name;
        let field_name = record_access.field;
        let span = record_access.span.clone();

        let definition = self
            .module
            .record(name.name)
            .ok_or_else(|| anyhow!("reference to undefined record '{}'", name))?;
        let field = definition
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| {
                anyhow!(
                    "reference to undefined field '{}' of record '{}'",
                    field_name,
                    name
                )
            })?;
        let field_var = self.next_var(Some(field_name.span));
        let catch_all_var = self.next_var(Some(span));

        let tuple_pattern = self.expand_record(&Record {
            span,
            name: name.clone(),
            fields: vec![RecordField {
                span: field_name.span,
                name: field.name,
                value: Some(Expr::Var(field_var.into())),
                ty: None,
            }],
        })?;
        // The callee for pattern match failure
        let erlang_error = Expr::FunctionName(FunctionName::Resolved(Span::new(
            span,
            syntax_core::FunctionName::new(symbols::Erlang, symbols::Error, 1),
        )));
        // The error reason
        let reason = Expr::Tuple(Tuple {
            span,
            elements: vec![
                Expr::Literal(Literal::Atom(Ident::with_empty_span(symbols::Badrecord))),
                Expr::Var(catch_all_var.into()),
            ],
        });
        // The expanded representation:
        // case Expr of
        //   {myrec, _, .., $0, ..} ->
        //     $0;
        //   $1 ->
        //     erlang:error({badrecord, $1})
        // end
        Ok(Expr::Case(Case {
            span: record_access.span.clone(),
            expr: record_access.record.clone(),
            clauses: vec![
                Clause {
                    span,
                    patterns: vec![tuple_pattern],
                    guards: vec![],
                    body: vec![Expr::Var(field_var.into())],
                    compiler_generated: true,
                },
                Clause {
                    span,
                    patterns: vec![Expr::Var(catch_all_var.into())],
                    guards: vec![],
                    body: vec![Expr::Apply(Apply {
                        span,
                        callee: Box::new(erlang_error),
                        args: vec![reason],
                    })],
                    compiler_generated: true,
                },
            ],
        }))
    }

    fn expand_update(&mut self, record_update: &RecordUpdate) -> anyhow::Result<Expr> {
        let name = record_update.name;
        let span = record_update.span.clone();

        let definition = self
            .module
            .record(name.name)
            .ok_or_else(|| anyhow!("reference to undefined record '{}'", name))?;

        // Save a copy of the record expression as we'll need that
        let expr = record_update.record.as_ref().clone();
        // If there are no updates for some reason, treat this expression as transparent
        if record_update.updates.is_empty() {
            return Ok(expr);
        }
        // Generate vars for use in the pattern match phase
        let bound_var = self.next_var(Some(span));
        let catch_all_var = self.next_var(Some(span));
        // Expand the updates to a sequence of nested setelement calls such that they evaluate in the correct order
        let expanded_updates = record_update
            .updates
            .iter()
            .rev()
            .try_fold::<_, _, anyhow::Result<Expr>>(
                Expr::Var(bound_var.into()),
                |acc, update| {
                    let field_name = update.name;
                    let position = definition
                        .fields
                        .iter()
                        .position(|f| f.name == field_name)
                        .ok_or_else(|| {
                            anyhow!(
                                "reference to undefined field '{}' of record '{}'",
                                field_name,
                                name
                            )
                        })?;

                    let callee = Expr::FunctionName(FunctionName::Resolved(Span::new(
                        span,
                        syntax_core::FunctionName::new(symbols::Erlang, symbols::Setelement, 3),
                    )));
                    let index = Expr::Literal(Literal::Integer(span, (position + 1).into()));
                    let value = update.value.as_ref().unwrap().clone();
                    Ok(Expr::Apply(Apply {
                        span,
                        callee: Box::new(callee),
                        args: vec![index, acc, value],
                    }))
                },
            )?;
        // Generate an empty pattern, i.e. all wildcards, that validates the input is a tuple of the appropriate type/shape
        let tuple_pattern = self.expand_record(&Record {
            span,
            name,
            fields: vec![],
        })?;
        // The callee for pattern match failure
        let erlang_error = Expr::FunctionName(FunctionName::Resolved(Span::new(
            span,
            syntax_core::FunctionName::new(symbols::Erlang, symbols::Error, 1),
        )));
        // The error reason
        let reason = Expr::Tuple(Tuple {
            span,
            elements: vec![
                Expr::Literal(Literal::Atom(Ident::with_empty_span(symbols::Badrecord))),
                Expr::Var(catch_all_var.into()),
            ],
        });

        Ok(Expr::Case(Case {
            span,
            expr: Box::new(expr),
            clauses: vec![
                Clause {
                    span,
                    patterns: vec![tuple_pattern],
                    guards: vec![],
                    body: vec![expanded_updates],
                    compiler_generated: true,
                },
                Clause {
                    span,
                    patterns: vec![Expr::Var(catch_all_var.into())],
                    guards: vec![],
                    body: vec![Expr::Apply(Apply {
                        span,
                        callee: Box::new(erlang_error),
                        args: vec![reason],
                    })],
                    compiler_generated: true,
                },
            ],
        }))
    }
}
