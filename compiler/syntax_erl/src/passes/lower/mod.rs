mod block;
mod builder;
mod exprs;
mod funs;
mod helpers;
mod patterns;
mod receive;
mod try_catch;

use liblumen_diagnostics::*;
use liblumen_intern::symbols;
use liblumen_pass::Pass;
use liblumen_syntax_core::*;
use log::debug;

use crate::ast;

use self::builder::IrBuilder;

/// This pass is responsible for transforming the processed AST to Core IR
pub struct AstToCore {
    reporter: Reporter,
}
impl AstToCore {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for AstToCore {
    type Input<'a> = ast::Module;
    type Output<'a> = Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut ir_module = Module::new(module.name);

        // Add all imports to the module
        for (_, sig) in module.imports.iter() {
            ir_module.import_function((**sig).clone());
        }

        // Declare all functions in the module, and store their refs so we can access them later
        let mut functions = Vec::with_capacity(module.functions.len());
        for (name, fun) in module.functions.iter() {
            let name = Spanned::new(fun.span, *name);
            let visibility = if module.exports.contains(&name) {
                Visibility::PUBLIC
            } else {
                Visibility::DEFAULT
            };
            let mut params = vec![];
            params.resize(fun.arity as usize, Type::Term(TermType::Any));
            let signature = Signature {
                visibility,
                cc: CallConv::Erlang,
                module: module.name(),
                name: fun.name.name,
                params,
                results: vec![
                    Type::Primitive(PrimitiveType::I1),
                    Type::Term(TermType::Any),
                ],
            };
            let id = ir_module.declare_function(signature.clone());
            debug!(
                "declared {} with visibility {}, it has id {:?}",
                signature.mfa(),
                signature.visibility,
                id
            );
            functions.push((id, signature));
        }

        // For every function in the module, run a function-local pass which produces the function body
        for (i, fun) in module.functions.into_values().enumerate() {
            let (id, sig) = functions.get(i).unwrap();
            let mut pass = LowerFunctionToCore {
                reporter: &mut self.reporter,
                module: &mut ir_module,
                id: *id,
                signature: sig.clone(),
                in_guard: false,
                landing_pads: vec![],
            };
            let ir_function = pass.run(fun)?;
            ir_module.define_function(ir_function);
        }

        debug!("successfully lowered syntax_erl module to syntax_core");
        Ok(ir_module)
    }
}

struct LowerFunctionToCore<'m> {
    reporter: &'m mut Reporter,
    module: &'m mut Module,
    id: FuncRef,
    signature: Signature,
    in_guard: bool,
    landing_pads: Vec<Block>,
}
impl<'m> Pass for LowerFunctionToCore<'m> {
    type Input<'a> = ast::Function;
    type Output<'a> = Function;

    fn run<'a>(&mut self, mut ast: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        assert_ne!(ast.clauses.len(), 0, "cannot lower empty function clauses");
        debug!(
            "running LowerFunctionToCore pass on {} ({:?})",
            self.signature.mfa(),
            self.id
        );

        // Construct an empty function which inherits its module context from the module
        // in which it was declared. The function definition itself must remain detached
        // from the module until this pass is complete, otherwise we end up with an issue
        // of nested mutable references. If at some point in the future, view types make
        // their way into Rust, we might be able to combine the two steps
        let mut function = Function::new(
            self.id,
            ast.span,
            self.signature.clone(),
            self.module.annotations.clone(),
            self.module.signatures.clone(),
            self.module.callees.clone(),
            self.module.constants.clone(),
        );
        let mut builder = IrBuilder::new(&mut function);

        // If there is only a single clause, then we can use the entry block for
        // lowering, since there is no risk of variable scope collisions
        let num_clauses = ast.clauses.len();
        debug!("num_clauses = {}", num_clauses);
        let reuse_entry = num_clauses < 2;
        debug!("reuse_entry = {}", reuse_entry);

        // Create blocks to act as the entry for each clause
        let clause_entries = {
            let mut blocks = Vec::with_capacity(ast.clauses.len());
            if reuse_entry {
                blocks.push((builder.entry, ast.clauses[0].span));
                for clause in ast.clauses.iter().skip(1) {
                    blocks.push((builder.create_block(), clause.span));
                }
            } else {
                for clause in ast.clauses.iter() {
                    blocks.push((builder.create_block(), clause.span));
                }
                // If we aren't reusing the entry block to avoid scoping conflicts,
                // then we require an unconditional branch in the entry block to the
                // first clause (i.e. clause entry)
                let (first_clause_entry, first_clause_span) = blocks[0];
                builder.ins().br(first_clause_entry, &[], first_clause_span);
                builder.switch_to_block(first_clause_entry);
            }
            blocks
        };

        for (i, clause) in ast.clauses.drain(0..).enumerate() {
            let (clause_entry, _) = clause_entries[i];
            let next_clause = clause_entries.get(i + 1).copied();
            self.lower_clause(&mut builder, clause, clause_entry, next_clause)?;
        }

        debug!("LowerFunctionToCore pass completed successfully");
        Ok(function)
    }
}

impl<'m> LowerFunctionToCore<'m> {
    fn lower_clause<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut clause: ast::FunctionClause,
        clause_entry: Block,
        next_clause: Option<(Block, SourceSpan)>,
    ) -> anyhow::Result<()> {
        use std::collections::HashSet;

        if builder.current_block() != clause_entry {
            builder.switch_to_block(clause_entry);
        }

        let function_params = builder.block_params(builder.entry).to_vec();

        // First, determine if we have any pattern matching occuring in the clause head,
        // and consequently, whether or not this is a wildcard clause.
        let mut has_implicit_match = false;
        let mut has_explicit_match = false;
        let mut has_guard = clause.guard.is_some();
        let mut clause_param_symbol_set = HashSet::with_capacity(clause.params.len());
        {
            for param in clause.params.iter() {
                if let Some(v) = param.as_var() {
                    let sym = v.sym();
                    if sym == symbols::WildcardMatch {
                        continue;
                    }
                    if !clause_param_symbol_set.insert(sym) {
                        has_implicit_match = true;
                    }
                } else {
                    has_explicit_match = true;
                }
            }
        }
        let is_wildcard = !has_implicit_match && !has_explicit_match && !has_guard;
        debug!("has_implicit_match = {}", has_implicit_match);
        debug!("has_explicit_match = {}", has_implicit_match);
        debug!("has_guard          = {}", has_implicit_match);
        debug!("is_wildcard        = {}", has_implicit_match);

        // If this clause has pattern matching in the head, we need to select an
        // appropriate landing pad for match failures. If there are multiple clauses,
        // then the next clause is used; if there are not, or this is the final clause,
        // then we generate a new block that raises a function_clause error.
        //
        // For wildcard clauses, no pattern fail block is needed
        let pattern_fail = if !is_wildcard {
            if let Some((next_clause_block, _)) = next_clause {
                // We need to trampoline exceptions to the next clause block
                // with a new block that simply drops the exception as it is ignored
                let current_block = builder.current_block();
                let trampoline = builder.create_block();
                let span = clause.span;
                builder.append_block_param(trampoline, Type::Term(TermType::Any), span);
                builder.ins().br(next_clause_block, &[], span);
                builder.switch_to_block(current_block);
                Some(next_clause_block)
            } else {
                // erlang:raise(error, function_clause, Trace).
                let current_block = builder.current_block();
                let pattern_fail_block = builder.create_block();
                // All landing pads require the exception argument
                let span = clause.span;
                builder.append_block_param(pattern_fail_block, Type::Term(TermType::Any), span);
                builder.switch_to_block(pattern_fail_block);
                // erlang:raise(error, function_clause, Trace).
                let class = builder.ins().atom(symbols::Error, span);
                let reason = builder.ins().atom(symbols::FunctionClause, span);
                let trace = builder.ins().build_stacktrace(span);
                builder.ins().raise(class, reason, trace, span);
                builder.switch_to_block(current_block);
                Some(pattern_fail_block)
            }
        } else {
            None
        };
        debug!("pattern_fail is {:?}", pattern_fail);

        // Next, for each function parameter, lower its pattern, if applicable.
        //
        // * If the param is simply a variable binding, and was not previously defined by another param, link it to the matching block param and define the var in scope
        // * If the param is a variable, but was previously defined by another param, it will be in scope, and when lowered as a pattern will be treated as an equality assertion
        // * If the param is any other kind of pattern, there are no vars to define yet, leave it as-is
        //
        // Once we have lowered the last pattern, the following should be true of the control flow::
        //
        // If the last pattern fails, control will jump unconditionally to the pattern_fail block
        // If the last pattern succeeds, and there is no guard, control proceeds with the clause body
        // If the last pattern succeeds, and there is a guard sequence, _and_ there is another clause following this one, use the next clause for guard failure, and start lowering the guards
        // If the last pattern succeeds, and there is a guard sequence, and this is the final clause, create a block which raises a function_clause error to use as the failure path, and start lowering the guards
        clause_param_symbol_set.clear();
        for (param, value) in clause
            .params
            .drain(0..)
            .zip(function_params.iter().copied())
        {
            if let Some(var) = param.as_var() {
                let sym = var.sym();
                if sym == symbols::WildcardMatch {
                    // There is nothing to bind here, this argument is ignored
                    continue;
                }
                if !clause_param_symbol_set.insert(sym) {
                    // This variable binding occurred earlier in the parameter list,
                    // so we need to lower it as a pattern
                    self.lower_pattern(
                        builder,
                        param,
                        value,
                        pattern_fail.expect("fallible patterns require a pattern_fail_block"),
                    )?;
                } else {
                    // This is a new variable binding, so define it in scope of the remaining params
                    builder.define_var(sym, value);
                }
            } else {
                self.lower_pattern(
                    builder,
                    param,
                    value,
                    pattern_fail.expect("fallible patterns require a pattern_fail_block"),
                )?;
            }
        }

        if is_wildcard {
            if let Some((_, next_clause_span)) = next_clause {
                self.show_error(
                    "unreachable clause",
                    &[
                        (next_clause_span, "this clause is unreachable"),
                        (clause.span, "because this clause shadows it"),
                    ],
                );
            } else {
                let last_expression_span = clause.body.last().map(|e| e.span()).unwrap();
                let (result, _) = self.lower_block(builder, clause.body)?;
                builder.ins().ret_ok(result, last_expression_span);
            }
            return Ok(());
        }

        // A guard sequence is a set of guards. A guard is a set of expressions.
        // Evaluation of a guard sequence can be described using the following
        // psuedocode:
        //
        //   for guard in guard_sequence:
        //     # A guard is true if all its expressions are truthy
        //     guard_passed = guard.all(|expr| expr() == true)
        //     # If the guard is true, then we can stop processing the guard sequence
        //     if guard_passed
        //       return true
        //   # If we reach the end of the sequence with no passing guards, the
        //   # entire sequence is considered failed
        //   return false
        let mut guard_sequence = clause.guard.take().unwrap();
        let guard_sequence_failed = pattern_fail.unwrap();
        let guard_sequence_passed = builder.create_block();
        let guard_blocks = guard_sequence
            .iter()
            .skip(1) // Skip the first block as we use the entry for it
            .map(|guard| {
                let guard_block = builder.create_block();
                builder.append_block_param(guard_block, Type::Term(TermType::Any), guard.span);
                guard_block
            })
            .collect::<Vec<_>>();
        for (i, guard) in guard_sequence.drain(0..).enumerate() {
            let guard_failed = guard_blocks
                .get(i + 1)
                .copied()
                .unwrap_or(guard_sequence_failed);
            self.lower_guard_sequence(builder, guard, guard_failed, guard_sequence_passed)?;
        }

        builder.switch_to_block(guard_sequence_passed);
        let last_expression_span = clause.body.last().map(|e| e.span()).unwrap();
        let (result, _) = self.lower_block(builder, clause.body)?;
        builder.ins().ret_ok(result, last_expression_span);

        Ok(())
    }

    fn lower_guard_sequence<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut sequence: ast::Guard,
        guard_failed: Block,
        guard_passed: Block,
    ) -> anyhow::Result<()> {
        assert!(
            !sequence.conditions.is_empty(),
            "guard sequences cannot be empty"
        );

        for guard in sequence.conditions.drain(0..) {
            self.lower_guard(builder, guard, guard_failed)?;
        }

        builder.ins().br(guard_passed, &[], sequence.span);
        builder.switch_to_block(guard_passed);
        Ok(())
    }

    fn lower_guard<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        guard: ast::Expr,
        guard_failed: Block,
    ) -> anyhow::Result<()> {
        self.in_guard = true;
        self.landing_pads.push(guard_failed);

        let span = guard.span();
        let result = self.lower_expr(builder, guard)?;

        let cond = builder
            .ins()
            .eq_exact_imm(result, Immediate::Bool(true), span);
        // TODO: This should probably be a match_fail error, but the value
        // is actually unused, so we simply use Term::None
        let exception = builder.ins().none(span);
        builder
            .ins()
            .br_unless(cond, guard_failed, &[exception], span);

        self.landing_pads.pop();
        self.in_guard = false;
        Ok(())
    }

    fn show_error(&mut self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::error().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }

    fn show_error_annotated(
        &mut self,
        message: &str,
        labels: &[(SourceSpan, &str)],
        notes: &[&str],
    ) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::error().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels)
                    .with_notes(notes.iter().map(|n| n.to_string()).collect()),
            );
        }
    }

    fn show_warning(&mut self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::warning().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }

    fn show_warning_annotated(
        &mut self,
        message: &str,
        labels: &[(SourceSpan, &str)],
        notes: &[&str],
    ) {
        if labels.is_empty() {
            self.reporter
                .diagnostic(Diagnostic::error().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.reporter.diagnostic(
                Diagnostic::warning()
                    .with_message(message)
                    .with_labels(labels)
                    .with_notes(notes.iter().map(|n| n.to_string()).collect()),
            );
        }
    }
}
