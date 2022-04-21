use std::collections::{BTreeMap, BTreeSet, HashSet};

use cranelift_entity::packed_option::ReservedValue;
use liblumen_intern::Symbol;
use liblumen_syntax_core::*;
use log::debug;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    /// Used to lower sequences of expressions which occur in a new lexical scope (function body, if branches)
    ///
    /// Erlang in general is quite standard with lexical scoping rules, with one weird exception. It supports
    /// something we generall call "imperative assignment", where bindings introduced in the nested scope of certain
    /// expressions are exported into containing scopes as long as the binding is introduced on all control flow
    /// paths of the expression. This scoping rule is allowed in `begin`, `if`, and `case`, even if it occurs during
    /// evaluation of a match binding (i.e `Foo = case .. of .. end, ..`).
    ///
    /// To deal with this, we track assignments made in the block, and return them along with the blocks result value.
    /// When lowering an expression that contains blocks, we re-export bindings from those blocks into the current scope,
    /// as long as those bindings are present in all control flow paths. This also requires us to pass all exported bindings
    /// as additional block arguments at the IR level, in order to ensure that when control flow joins occur, that the bindings
    /// have the correct value associated with them. When exporting variables, we use the values of those block arguments, as
    /// the use of those bindings will occur in successor blocks.
    pub(super) fn lower_block<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut exprs: Vec<ast::Expr>,
    ) -> anyhow::Result<(Value, BTreeMap<Symbol, Value>)> {
        debug!(
            "lower block (is_terminated = {})",
            builder.is_current_block_terminated()
        );
        // Holds bindings which should be exported from this block
        // NOTE: It is the responsibility of the caller to handle these
        // exports as appropriate from the caller context. It is fine to
        // ignore them, as not all expressions permit exported bindings,
        // e.g. `receive` or `try`
        let mut exports = BTreeMap::new();
        // Blocks are themselves expressions, so this holds the value which
        // is to be returned as the value of the block
        let mut block_result = Value::reserved_value();

        // We handle all expression types here which are valid at the top-level
        // of a block, but for expressions which have no block-like structure,
        // we delegate to `lower_expr` instead. This function is primarily concerned
        // with those block-like expressions
        let last_expr_index = exprs.len() - 1;
        for (i, expr) in exprs.drain(0..).enumerate() {
            let is_last = i == last_expr_index;
            debug!("lowering block for ast expression {:?}", &expr);
            match expr {
                ast::Expr::Var(var) if is_last => {
                    block_result = self.lower_var(builder, var)?;
                }
                ast::Expr::Literal(lit) if is_last => {
                    block_result = self.lower_literal(builder, lit)?;
                }
                ast::Expr::FunctionName(_) | ast::Expr::DelayedSubstitution(_, _) => {
                    panic!(
                        "unexpected expression type found during lowering of block: {:?}",
                        &expr
                    );
                }
                ast::Expr::Nil(nil) if is_last => {
                    block_result = builder.ins().nil(nil.span());
                }
                ast::Expr::Cons(cons) if is_last => {
                    block_result = self.lower_cons(builder, cons)?;
                }
                ast::Expr::Tuple(tuple) if is_last => {
                    block_result = self.lower_tuple(builder, tuple)?;
                }
                ast::Expr::Map(map) if is_last => {
                    block_result = self.lower_map(builder, map)?;
                }
                ast::Expr::MapUpdate(map_update) if is_last => {
                    block_result = self.lower_map_update(builder, map_update)?;
                }
                ast::Expr::MapProjection(proj) => panic!(
                    "unexpected expression type found during lowering of block: {:?}",
                    &proj
                ),
                ast::Expr::Binary(bin) if is_last => {
                    block_result = self.lower_binary(builder, bin)?;
                }
                ast::Expr::Record(_)
                | ast::Expr::RecordAccess(_)
                | ast::Expr::RecordIndex(_)
                | ast::Expr::RecordUpdate(_) => {
                    panic!(
                        "unexpected expression type found during lowering of block: {:?}",
                        &expr
                    );
                }
                ast::Expr::ListComprehension(lc) if is_last => {
                    block_result = self.lower_lc(builder, lc)?;
                }
                ast::Expr::BinaryComprehension(bc) if is_last => {
                    block_result = self.lower_bc(builder, bc)?;
                }
                ast::Expr::Generator(_) | ast::Expr::BinaryGenerator(_) => panic!(
                    "unexpected expression type found during lowering of block: {:?}",
                    &expr
                ),
                // This is really just a nested set of expressions, so we handle it inline
                ast::Expr::Begin(begin) => {
                    let (result, exported) = self.lower_block(builder, begin.body)?;
                    for (binding, value) in exported.iter() {
                        debug_assert!(!exports.contains_key(binding), "begin block exported a binding for '{}' when it should have been treated as a match", binding);
                        exports.insert(*binding, *value);
                    }
                    block_result = result;
                }
                // Function calls can have side-effects, so they are always lowered, even if the result is unused
                ast::Expr::Apply(apply) => {
                    block_result = self.lower_apply(builder, apply, is_last)?;
                }
                ast::Expr::Remote(remote) => panic!(
                    "unexpected expression type found during lowering of block: {:?}",
                    &remote
                ),
                ast::Expr::BinaryExpr(op) => {
                    block_result = self.lower_binary_op(builder, op)?;
                }
                ast::Expr::UnaryExpr(op) => {
                    block_result = self.lower_unary_op(builder, op)?;
                }
                // This is pretty much like the equivalent branch in `lower_expr`, but we specifically handle
                // the case where the bound expression is block-like
                ast::Expr::Match(match_expr) => {
                    block_result = self.lower_match_block(builder, match_expr, &mut exports)?;
                }
                // With ifs, things are complicated, as we must unify the exports from every branch, and each
                // branch of the if can itself be quite complex
                ast::Expr::If(if_expr) => {
                    block_result = self.lower_if_block(builder, if_expr, &mut exports)?;
                }
                ast::Expr::Catch(catch) => {
                    block_result = self.lower_catch(builder, catch)?;
                }
                // Likewise with case
                ast::Expr::Case(case) => {
                    block_result = self.lower_case_block(builder, case, &mut exports)?;
                }
                // Receives do not export bindings, but they have a complex state machine representation
                ast::Expr::Receive(receive) => {
                    block_result = self.lower_receive(builder, receive)?;
                }
                // Try expressions also do not export bindings, but they have several different configurations
                // possible, so they are somewhat complex as well
                ast::Expr::Try(try_expr) => {
                    block_result = self.lower_try(builder, try_expr)?;
                }
                // Funs are complicated for a variety of reasons, but we treat them as regular expressions,
                // as they don't export variables into their containing scope
                ast::Expr::Fun(fun) if is_last => {
                    block_result = self.lower_fun(builder, fun)?;
                }
                // At this point, what remains are expressions which construct terms that are never used,
                // so we warn about it and skip lowering them
                ast::Expr::Var(var) => {
                    debug_assert!(!is_last);
                    let span = var.span();
                    self.show_warning(
                        "expression is meaningless",
                        &[(span, "the value of this expression is never used")],
                    );
                }
                ast::Expr::Literal(_)
                | ast::Expr::Nil(_)
                | ast::Expr::Cons(_)
                | ast::Expr::Tuple(_)
                | ast::Expr::Map(_)
                | ast::Expr::MapUpdate(_)
                | ast::Expr::Binary(_)
                | ast::Expr::ListComprehension(_)
                | ast::Expr::BinaryComprehension(_)
                | ast::Expr::Fun(_) => {
                    let span = expr.span();
                    self.show_warning_annotated(
                        "term is constructed, but never used",
                        &[(span, "the value of this expression is never used")],
                        &["This warning occurred because the expression is not the last expression in its block, and is not bound to a variable, so its value cannot be used. Consider removing this expression."]
                    );
                }
            }
        }

        debug_assert!(
            !builder.is_current_block_terminated(),
            "you must leave the current block open when lowering blocks"
        );

        Ok((block_result, exports))
    }

    pub(super) fn lower_match_block<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: ast::Match,
        exports: &mut BTreeMap<Symbol, Value>,
    ) -> anyhow::Result<Value> {
        let pattern = *expr.pattern;
        let input = *expr.expr;
        // Lower the value, and if the value is block-like, handle re-exporting its exports
        let value = if input.is_block_like() {
            let (result, exported) = self.lower_block_expr(builder, input)?;
            for (bound, value) in exported.iter() {
                debug_assert!(
                    !exports.contains_key(bound),
                    "leaked export which should have been lowered as pattern match: {:?}",
                    bound
                );
                let bound = *bound;
                let value = *value;
                builder.define_var(bound, value);
                exports.insert(bound, value);
            }
            result
        } else {
            self.lower_expr(builder, input)?
        };
        // Perform the match/bind new variable
        match pattern {
            ast::Expr::Var(var) => {
                let sym = var.sym();
                if let Some(_bound) = builder.get_var(sym) {
                    // This variable is bound in scope, so this is actually an
                    // equality assertion
                    todo!("pattern matching against previously bound vars with '='")
                } else {
                    // This variable is not bound in scope, so add a new binding
                    builder.define_var(sym, value);
                    Ok(value)
                }
            }
            _ => todo!("pattern matches using arbitrary expressions with '='"),
        }
    }

    /// Lowering an if block is a bit of a mess.
    ///
    /// We're essentially transforming this into a sequence of if/then conditionals,
    /// but we also need to ensure that if imperative assignment occurs in the body of
    /// any of the branches, that we re-export those bindings into the containing scope
    /// if the bindings are common across all branches.
    ///
    /// If conditionals must be guard-safe expressions, which simplifies things a bit,
    /// and they must return exactly `true` to pass the guard, otherwise the conditional
    /// is false.
    ///
    /// The transformation we perform here is, as follows:
    ///
    /// * Create a block for the output (i.e. where control flow joins after evaluating the if).
    ///   This block requires one block argument for the result, plus N block arguments for exported
    ///   variables
    /// * If there is no wildcard clause (i.e. the final pattern is a literal `true`), create a block to handle the match failure
    ///   This block should use the match_fail primop, with an argument of `if_clause`
    /// * Starting with the first clause, lower the clause expression and then perform a conditional branch to the approrpriate target block
    /// * After lowering, switch to the output block, and set the result value for this lowering to the first block argument, i.e. the
    /// result of evaluating the `if`
    pub(super) fn lower_if_block<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut expr: ast::If,
        exports: &mut BTreeMap<Symbol, Value>,
    ) -> anyhow::Result<Value> {
        let span = expr.span;
        let current_block = builder.current_block();
        let has_wildcard = expr.has_wildcard_clause();
        let output_block = builder.create_block();
        let output_result =
            builder.append_block_param(output_block, Type::Term(TermType::Any), span);
        // Ensure the variable scope for the output block is the same scope as the expression
        builder.set_scope(output_block, builder.get_scope(current_block));

        // Create match fail clause if no wildcard is present, use the wildcard clause otherwise
        let match_fail_block = if has_wildcard {
            None
        } else {
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder
                .ins()
                .match_fail_imm(Immediate::Atom(symbols::IfClause), span);
            builder.switch_to_block(current_block);
            Some(block)
        };

        let last_clause_index = expr.clauses.len() - 1;
        let mut wildcard_span = None;
        let mut clause_exports = Vec::with_capacity(expr.clauses.len());
        for (i, clause) in expr.clauses.drain(0..).enumerate() {
            let is_last = i == last_clause_index;
            let clause_span = clause.span;
            // Print error for all unmatchable clauses
            if let Some(prev_span) = wildcard_span {
                self.show_error(
                    "unreachable code",
                    &[
                        (clause_span, "this clause is unreachable"),
                        (prev_span, "because this clause always matches"),
                    ],
                );
                continue;
            }
            // If this is a wildcard clause, it always executes, so we lower its
            // block and use the resulting value in our unconditional branch to
            // the output block
            if clause.is_wildcard() {
                let (value, exported) = self.lower_block(builder, clause.body)?;
                let inst = builder.ins().br(output_block, &[value], clause_span);
                // Save the branch instruction, the block it's in, and the exports from the associated clause
                clause_exports.push((builder.current_block(), inst, exported));
                // If this isn't the last clause, trigger all remaining clauses
                // to be skipped and raise errors indicating they're unreachable
                if !is_last {
                    wildcard_span = Some(clause_span);
                }
                continue;
            }

            // Otherwise, we need to create a block for the next clause, or use the match_fail
            // block, depending on whether there is a wildcard clause or not. We then lower the
            // guard sequence using that block as the failure target for the guard.
            let next_clause = if is_last {
                match_fail_block.expect("should have created a match_fail block")
            } else {
                builder.create_block()
            };
            let mut guard_sequence = clause.guards;
            // If the entire guard sequence fails, we either jump to the next clause or the match_fail block
            let guard_sequence_failed = next_clause;
            // If the sequence passes, we need a block in which to lower the clause body
            let guard_sequence_passed = builder.create_block();
            // Each guard gets its own block to keep things simple conceptually
            // When a guard fails, its failure target is either the next guard, or the overall sequence failure block
            let guard_blocks = guard_sequence
                .iter()
                .map(|_| builder.create_block())
                .collect::<Vec<_>>();
            for (i, guard) in guard_sequence.drain(0..).enumerate() {
                let guard_failed = guard_blocks
                    .get(i + 1)
                    .copied()
                    .unwrap_or(guard_sequence_failed);
                self.lower_guard_sequence(builder, guard, guard_failed, guard_sequence_passed)?;
            }
            // After lowering the guard sequence, we need to lower the clause body, and from there, branch to the output block
            builder.switch_to_block(guard_sequence_passed);
            let last_expression_span = clause.body.last().map(|e| e.span()).unwrap();
            let (result, exported) = self.lower_block(builder, clause.body)?;
            // Save the branch instruction, the block it's in, and the exports from the associated clause
            let inst = builder
                .ins()
                .br(output_block, &[result], last_expression_span);
            clause_exports.push((builder.current_block(), inst, exported));
            // If we have another clause to lower, we need to switch to the next block
            if !is_last {
                builder.switch_to_block(next_clause);
            }
        }

        // We now need to unify the common exports across all the clauses, update the
        // block arguments for the output block to match the exports, and rewrite
        // the branch instruction from each clause block to append the extra arguments
        // in the jump to the output block. Lastly, we need to define exports in scope
        // and re-export them to this expression's parent scope

        // Gather the set of common exports from each branch, sorted by symbol
        let common_exports = clause_exports
            .iter()
            .fold(None, |acc: Option<BTreeSet<Symbol>>, (_, _, exported)| {
                if let Some(prev) = acc {
                    let a: HashSet<Symbol> = prev.iter().copied().collect();
                    let b: HashSet<Symbol> = exported.keys().copied().collect();
                    Some(a.intersection(&b).copied().collect::<BTreeSet<_>>())
                } else {
                    Some(exported.keys().copied().collect::<BTreeSet<_>>())
                }
            })
            .unwrap();
        // Track the output block values for each export to be used in a later step
        let mut export_values = Vec::with_capacity(common_exports.len());
        // Append extra block parameters for each exported var
        for _ in common_exports.iter() {
            export_values.push(builder.append_block_param(
                output_block,
                Type::Term(TermType::Any),
                span,
            ));
        }
        // Rewrite the br instruction arguments from each clause body to append the exported values
        let mut args = Vec::with_capacity(common_exports.len());
        for (_, br_inst, exported) in clause_exports.iter() {
            args.clear();
            for export in common_exports.iter() {
                args.push(exported.get(export).copied().unwrap());
            }
            builder.append_inst_args(*br_inst, args.as_slice());
        }
        // Bind exported values in the scope of the output block, and re-export them to the parent scope
        for (name, value) in common_exports
            .iter()
            .copied()
            .zip(export_values.iter().copied())
        {
            // Define binding in current scope
            builder.define_var(name, value);
            // Re-export to parent scope
            exports.insert(name, value);
        }

        // Leave the builder in the output block
        builder.switch_to_block(output_block);

        Ok(output_result)
    }

    pub(super) fn lower_case_block<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _case: ast::Case,
        _exports: &mut BTreeMap<Symbol, Value>,
    ) -> anyhow::Result<Value> {
        todo!()
    }
}
