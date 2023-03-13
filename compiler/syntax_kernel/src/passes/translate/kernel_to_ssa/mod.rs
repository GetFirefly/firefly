use std::assert_matches::assert_matches;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;

use log::debug;

use rpds::Stack;

use smallvec::SmallVec;

use firefly_binary::BinaryEntrySpecifier;
use firefly_diagnostics::*;
use firefly_intern::{symbols, Symbol};
use firefly_number::Int;
use firefly_pass::Pass;
use firefly_syntax_base::*;
use firefly_syntax_ssa::*;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::ir::{self as k, Expr as KExpr};

mod builder;
use self::builder::IrBuilder;

/// This pass is responsible for transforming the processed Kernel IR to SSA IR for code generation
pub struct KernelToSsa {
    diagnostics: Arc<DiagnosticsHandler>,
}
impl KernelToSsa {
    pub fn new(diagnostics: Arc<DiagnosticsHandler>) -> Self {
        Self { diagnostics }
    }
}
impl Pass for KernelToSsa {
    type Input<'a> = k::Module;
    type Output<'a> = Module;

    fn run<'a>(&mut self, mut module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut ir_module = Module::new(module.name);

        // Declare all functions in the module, and store their refs so we can access them later
        let mut functions = Vec::with_capacity(module.functions.len());
        for kfunction in module.functions.iter() {
            let name = Span::new(kfunction.span(), kfunction.name);
            let base_visibility = if module.exports.contains(&name) {
                Visibility::PUBLIC
            } else {
                Visibility::DEFAULT
            };
            let visibility = if kfunction.has_annotation(symbols::Nif) {
                base_visibility | Visibility::NIF
            } else {
                base_visibility
            };
            let params = vec![Type::Term(TermType::Any); name.arity as usize];
            let signature = Signature {
                visibility,
                cc: CallConv::Erlang,
                module: module.name.name,
                name: kfunction.name.function,
                ty: FunctionType::new(params, vec![Type::Term(TermType::Any)]),
            };
            let id = if kfunction.has_annotation(symbols::Closure) {
                ir_module.declare_closure(signature.clone())
            } else {
                ir_module.declare_function(signature.clone())
            };
            debug!(
                "declared {} with visibility {}, it has id {:?}",
                signature.mfa(),
                signature.visibility,
                id
            );
            functions.push((id, signature));
        }

        // For every function in the module, run a function-local pass which produces the function
        // body
        for (i, function) in module.functions.drain(..).enumerate() {
            let (id, sig) = functions.get(i).unwrap();
            let mut pass = LowerFunctionToSsa {
                diagnostics: &self.diagnostics,
                module: &mut ir_module,
                id: *id,
                signature: sig.clone(),
                labels: HashMap::new(),
                landing_pads: vec![],
                fail: Block::default(),
                brk: vec![],
                recv: Stack::new(),
            };
            let ir_function = pass.run(function)?;
            ir_module.define_function(ir_function);
        }

        debug!("successfully lowered kernel module to core ir module");
        Ok(ir_module)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FailContext {
    // In a function body, not in the scope of try/catch or catch
    Uncaught(Block),
    // In the scope of a try/catch or catch
    Catch(Block),
    // Inside a guard
    Guard(Block),
}
impl FailContext {
    #[allow(unused)]
    pub fn block(&self) -> Block {
        match self {
            Self::Uncaught(blk) | Self::Catch(blk) | Self::Guard(blk) => *blk,
        }
    }
}

struct LowerFunctionToSsa<'m> {
    diagnostics: &'m DiagnosticsHandler,
    module: &'m mut Module,
    id: FuncRef,
    signature: Signature,
    labels: HashMap<Symbol, Block>,
    landing_pads: Vec<Block>,
    fail: Block,
    // The current break label stack
    brk: Vec<Block>,
    // The current receive label stack
    #[allow(dead_code)]
    recv: Stack<Block>,
}
impl<'m> Pass for LowerFunctionToSsa<'m> {
    type Input<'a> = k::Function;
    type Output<'a> = Function;

    fn run<'a>(&mut self, kfunction: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        debug!(
            "running LowerFunctionToSsa pass on {} ({:?})",
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
            kfunction.span(),
            self.signature.clone(),
            self.module.signatures.clone(),
            self.module.callees.clone(),
            self.module.constants.clone(),
        );
        let mut builder = IrBuilder::new(&mut function);

        // Define the function parameters in the entry block and associate the CST vars
        // with those IR values
        let entry = builder.current_block();
        let var_types = kfunction
            .vars
            .iter()
            .zip(builder.func.signature.params().iter().cloned())
            .map(|(v, ty)| (v.span(), v.name(), ty))
            .collect::<SmallVec<[(SourceSpan, Symbol, Type); 8]>>();
        for (span, var, ty) in var_types {
            let value = builder.append_block_param(entry, ty, span);
            builder.define_var(var, value);
        }

        self.lower(&mut builder, *kfunction.body)?;

        // Prune any unreachable blocks generated due to the structure of Kernel Erlang
        // builder.prune_unreachable_blocks();

        debug!("LowerFunctionToSsa pass completed successfully");
        Ok(function)
    }
}
impl<'m> LowerFunctionToSsa<'m> {
    fn lower<'a>(&mut self, builder: &'a mut IrBuilder, expr: KExpr) -> anyhow::Result<()> {
        match expr {
            KExpr::Match(k::Match {
                span,
                box body,
                ret,
                ..
            }) => {
                let brk = builder.create_block();
                for v in ret.iter().map(|e| e.as_var().unwrap()) {
                    let value = builder.append_block_param(brk, Type::Term(TermType::Any), span);
                    builder.define_var(v.name(), value);
                }
                self.brk.push(brk);
                self.lower_match(builder, self.fail, body)?;
                self.brk.pop();
                // If the break block we created remains empty and
                // there are no values returned from this match, then
                // we know that this block is useless and can be removed
                if ret.is_empty() && builder.is_block_empty(brk) {
                    builder.remove_block(brk);
                    return Ok(());
                }
                // Otherwise, the values returned from the match are used
                // and we have more instructions to generate starting in
                // the break block
                builder.switch_to_block(brk);
                Ok(())
            }
            KExpr::If(expr) => self.lower_if(builder, expr),
            KExpr::Seq(k::Seq {
                box arg, box body, ..
            }) => {
                self.lower(builder, arg)?;
                self.lower(builder, body)
            }
            KExpr::Call(call) => self.lower_call(builder, call),
            KExpr::Enter(enter) => self.lower_enter(builder, enter),
            KExpr::Bif(bif) => self.lower_bif(builder, bif),
            KExpr::Try(expr) => self.lower_try(builder, expr),
            KExpr::TryEnter(expr) => self.lower_try_enter(builder, expr),
            KExpr::Catch(expr) => self.lower_catch(builder, expr),
            KExpr::Put(put) => self.lower_put(builder, put),
            KExpr::Return(k::Return { span, mut args, .. }) => {
                assert_eq!(args.len(), 1);
                let value = self.ssa_value(builder, args.pop().unwrap())?;
                if builder.is_current_block_terminated() {
                    // We may generate redundant return expressions due to exception bifs, elide
                    // them
                    Ok(())
                } else {
                    builder.ins().ret(value, span);
                    Ok(())
                }
            }
            KExpr::Break(k::Break { span, args, .. }) => {
                let args = self.ssa_values(builder, args)?;
                if builder.is_current_block_terminated() {
                    // We may generate redundant break expressions due to exception bifs, elide them
                    assert_eq!(args.len(), 1);
                    Ok(())
                } else {
                    let brk = self.brk.last().copied().expect("break target is missing");
                    builder.ins().br(brk, args.as_slice(), span);
                    Ok(())
                }
            }
            KExpr::LetRecGoto(k::LetRecGoto {
                label,
                vars,
                box first,
                box then,
                ret,
                ..
            }) => {
                let then_block = builder.create_block();
                for v in vars.iter() {
                    let value =
                        builder.append_block_param(then_block, Type::Term(TermType::Any), v.span());
                    builder.define_var(v.name(), value);
                }
                let final_block = builder.create_block();
                for v in ret.iter().map(|e| e.as_var().map(|v| v.name).unwrap()) {
                    let value = builder.append_block_param(
                        final_block,
                        Type::Term(TermType::Any),
                        v.span(),
                    );
                    builder.define_var(v.name, value);
                }
                self.labels.insert(label, then_block);
                self.brk.push(final_block);
                self.lower(builder, first)?;
                builder.switch_to_block(then_block);
                self.lower(builder, then)?;
                self.labels.remove(&label);
                self.brk.pop();
                builder.switch_to_block(final_block);
                Ok(())
            }
            KExpr::Goto(k::Goto {
                span, label, args, ..
            }) => {
                let target = *self.labels.get(&label).expect("goto target is missing");
                let args = self.ssa_values(builder, args)?;
                builder.ins().br(target, args.as_slice(), span);
                Ok(())
            }
            expr => panic!("unexpected expression type in call to lower: {:#?}", &expr),
        }
    }

    fn lower_if<'a>(&mut self, builder: &'a mut IrBuilder, expr: k::If) -> anyhow::Result<()> {
        let span = expr.span();
        let cond = self.ssa_value(builder, *expr.cond)?;
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let final_block = builder.create_block();
        for v in expr.ret.iter().map(|e| e.as_var().unwrap()) {
            let value = builder.append_block_param(final_block, Type::Term(TermType::Any), span);
            builder.define_var(v.name(), value);
        }
        builder
            .ins()
            .cond_br(cond, then_block, &[], else_block, &[], span);
        builder.switch_to_block(then_block);
        self.brk.push(final_block);
        self.lower(builder, *expr.then_body)?;
        builder.switch_to_block(else_block);
        self.lower(builder, *expr.else_body)?;
        builder.switch_to_block(final_block);
        self.brk.pop();
        Ok(())
    }

    ///  Generate code for a match tree.
    fn lower_match<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        fail: Block,
        body: KExpr,
    ) -> anyhow::Result<()> {
        match body {
            KExpr::Alt(k::Alt {
                box first,
                box then,
                ..
            }) => {
                let then_blk = builder.create_block();
                self.lower_match(builder, then_blk, first)?;
                builder.switch_to_block(then_blk);
                self.lower_match(builder, fail, then)
            }
            KExpr::Select(k::Select {
                span,
                ref var,
                mut types,
                ..
            }) => {
                let mut blocks = types
                    .iter()
                    .skip(1)
                    .map(|_| builder.create_block())
                    .collect::<Vec<_>>();
                blocks.push(fail);
                for (ty, block) in types.drain(..).zip(blocks.drain(..)) {
                    self.lower_select(builder, span, var, ty, block, fail)?;
                    builder.switch_to_block(block);
                }
                Ok(())
            }
            KExpr::Guard(k::Guard { mut clauses, .. }) => {
                let mut blocks = clauses
                    .iter()
                    .skip(1)
                    .map(|_| builder.create_block())
                    .collect::<Vec<_>>();
                blocks.push(fail);
                for (clause, block) in clauses.drain(..).zip(blocks.drain(..)) {
                    self.lower_guard(builder, clause, block)?;
                    builder.switch_to_block(block);
                }
                Ok(())
            }
            body => self.lower(builder, body),
        }
    }

    /// Selecting type and value needs two failure labels, `type_fail` is the
    /// block to jump to of the next type test when this type fails, and
    /// `value_fail` is the block when this type is correct but the value is
    /// wrong.  These are different as in the second case there is no need
    /// to try the next type, it will always fail.
    fn lower_select<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        mut clause: k::TypeClause,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        use crate::ir::MatchType;

        match clause.ty {
            MatchType::Binary if clause.values.len() == 1 => {
                let clause = clause.values.pop().unwrap();
                self.select_binary(builder, span, var, clause, type_fail, value_fail)
            }
            MatchType::BinarySegment | MatchType::BinaryInt => {
                self.select_binary_segments(builder, span, var, clause.values, type_fail)
            }
            MatchType::BinaryEnd if clause.values.len() == 1 => {
                let clause = clause.values.pop().unwrap();
                self.select_binary_end(builder, span, var, clause, type_fail)
            }
            MatchType::Map => {
                self.select_map(builder, span, var, clause.values, type_fail, value_fail)
            }
            MatchType::Cons if clause.values.len() == 1 => {
                let clause = clause.values.pop().unwrap();
                self.select_cons(builder, span, var, clause, type_fail, value_fail)
            }
            MatchType::Nil if clause.values.len() == 1 => {
                let clause = clause.values.pop().unwrap();
                self.select_nil(builder, span, var, clause, type_fail, value_fail)
            }
            MatchType::Literal => {
                self.select_literal(builder, span, var, clause.values, type_fail, value_fail)
            }
            MatchType::Tuple => {
                // Value clauses should have differing arity at this stage, clauses with same
                // arity are necessarily shadowed by the first clause. Our job here is to verify
                // this, and order the clauses by arity, then lower this match based on a type
                // guard and dispatch on arity
                let mut clauses = clause
                    .values
                    .drain(..)
                    .map(|vclause| {
                        let arity = match vclause.value.as_ref() {
                            KExpr::Tuple(t) => t.elements.len() as u32,
                            other => panic!("expected tuple expression here, got: {:#?}", other),
                        };
                        (arity, vclause)
                    })
                    .collect::<Vec<_>>();
                clauses.sort_by_key(|(arity, _)| *arity);
                let mut prev = None;
                for (arity, clause) in clauses.iter() {
                    match prev {
                        Some(prev_arity) if arity == prev_arity => {
                            panic!(
                                "found duplicate select clause for arity {}: {:#?}",
                                arity, clause
                            );
                        }
                        None | Some(_) => {
                            prev = Some(arity);
                            continue;
                        }
                    }
                }
                // Create a block for each combined set of values
                let mut blocks = clauses
                    .iter()
                    .map(|_| builder.create_block())
                    .collect::<Vec<_>>();
                let src = builder.var(var.name()).unwrap();
                // Tuples require us to do a type check for tuple; then do a dispatch
                // based on the arity of the tuple, so we use a fused instruction for this
                let (is_tuple, arity) = builder.ins().is_tuple_fetch_arity(src, span);
                builder.ins().br_unless(is_tuple, type_fail, &[], span);
                // The source value is known to be a tuple, so perform a cast before proceeding
                let src = builder
                    .ins()
                    .cast(src, Type::Term(TermType::Tuple(None)), span);
                // Dispatch on the arity to the appropriate block for that clause
                let arms = clauses
                    .iter()
                    .map(|(arity, _)| *arity)
                    .zip(blocks.iter().copied())
                    .collect::<Vec<_>>();
                builder.ins().switch(arity, arms, value_fail, span);
                // Now, for each clause, lower the body of that clause in the appropriate block
                for ((_, clause), block) in clauses.drain(..).zip(blocks.drain(..)) {
                    builder.switch_to_block(block);
                    // Bind tuple elemnts in this block
                    let KExpr::Tuple(tuple) = *clause.value else { unreachable!() };
                    for (i, elem) in tuple.elements.iter().enumerate() {
                        if elem.has_annotation(symbols::Unused) {
                            continue;
                        }
                        let var = elem.as_var().map(|v| v.name).unwrap();
                        let elem = builder.ins().get_element_imm(src, i, var.span());
                        builder.define_var(var.name, elem);
                    }
                    self.lower_match(builder, value_fail, *clause.body)?;
                }
                Ok(())
            }
            ty @ (MatchType::Atom | MatchType::Float | MatchType::Int) => {
                // Create a block for each value clause
                let mut blocks = clause
                    .values
                    .iter()
                    .map(|_| builder.create_block())
                    .collect::<Vec<_>>();
                let src = builder.var(var.name()).unwrap();
                let current_block = builder.current_block();
                // Generate type test
                let is_type = match ty {
                    MatchType::Atom => builder.ins().is_type(Type::Term(TermType::Atom), src, span),
                    MatchType::Float => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Float), src, span)
                    }
                    MatchType::Int => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Integer), src, span)
                    }
                    _ => unreachable!(),
                };
                // Jump to next type if the type test fails
                builder.ins().br_unless(is_type, type_fail, &[], span);
                // Lower each value test
                for (vclause, block) in clause.values.drain(..).zip(blocks.drain(..)) {
                    let span = vclause.span();
                    let val = self.lower_literal(builder, vclause.value.into_literal().unwrap())?;
                    let is_eq = builder.ins().eq_exact(src, val, span);
                    builder.ins().br_if(is_eq, block, &[], span);
                    builder.switch_to_block(block);
                    self.lower_match(builder, value_fail, *vclause.body)?;
                    builder.switch_to_block(current_block);
                }
                // If no test succeeds, branch to the value_fail block
                builder.ins().br(value_fail, &[], span);
                Ok(())
            }
            ty => panic!("unexpected match type: {:#?}", &ty),
        }
    }

    /// A guard is a boolean expression of tests.  Tests return true or
    /// false.  A fault in a test causes the test to return false.  Tests
    /// never return the boolean, instead we generate jump code to go to
    /// the correct exit point.  Primops and tests all go to the next
    /// instruction on success or jump to a failure block.
    fn lower_guard<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        clause: k::GuardClause,
        fail: Block,
    ) -> anyhow::Result<()> {
        self.lower_guard_expr(builder, fail, *clause.guard)?;
        self.lower_match(builder, fail, *clause.body)
    }

    fn lower_guard_expr<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        fail: Block,
        guard: KExpr,
    ) -> anyhow::Result<()> {
        match guard {
            KExpr::Try(k::Try {
                span,
                box arg,
                vars,
                body: box KExpr::Break(bbrk),
                evars,
                handler: box KExpr::Break(hbrk),
                ..
            }) if vars.is_empty()
                && evars.is_empty()
                && bbrk.args.is_empty()
                && hbrk.args.is_empty() =>
            {
                // Do a try/catch without return value for effect. The return
                // value is not checked; success passes on to the next instruction
                // and failure jumps to Fail.
                let final_block = builder.create_block();
                let old_fail = self.fail;
                self.fail = fail;
                self.brk.push(final_block);
                builder.ins().start_catch(fail, span);
                self.lower_guard_expr(builder, fail, arg)?;
                self.brk.pop();
                self.fail = old_fail;
                builder.switch_to_block(final_block);
                builder.ins().end_catch(span);
                Ok(())
            }
            KExpr::Test(k::Test { span, op, args, .. }) => {
                self.lower_test(builder, span, op, args, fail)
            }
            KExpr::Seq(k::Seq {
                box arg, box body, ..
            }) => {
                self.lower_guard_expr(builder, fail, arg)?;
                self.lower_guard_expr(builder, fail, body)
            }
            guard => self.lower(builder, guard),
        }
    }

    fn lower_test<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        op: FunctionName,
        mut args: Vec<KExpr>,
        fail: Block,
    ) -> anyhow::Result<()> {
        match (op.function, args.as_slice()) {
            (
                symbols::IsRecord,
                [_tuple, KExpr::Literal(Literal {
                    value: Lit::Atom(tag),
                    ..
                }), KExpr::Literal(Literal {
                    value: Lit::Integer(arity),
                    ..
                })],
            ) => {
                let tag = *tag;
                let arity = arity.to_usize().unwrap();
                let tuple = self.ssa_value(builder, args.remove(0))?;
                self.lower_test_is_record(builder, span, tuple, tag, arity, fail)
            }
            _ if op.is_type_test() => {
                let arg = self.ssa_value(builder, args.pop().unwrap())?;
                let result = match op.function {
                    symbols::IsAtom => builder.ins().is_type(Type::Term(TermType::Atom), arg, span),
                    symbols::IsBinary => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Binary), arg, span)
                    }
                    symbols::IsBitstring => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Bitstring), arg, span)
                    }
                    symbols::IsBoolean => {
                        builder.ins().is_type(Type::Term(TermType::Bool), arg, span)
                    }
                    symbols::IsFloat => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Float), arg, span)
                    }
                    symbols::IsInteger => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Integer), arg, span)
                    }
                    symbols::IsList => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::List(None)), arg, span)
                    }
                    symbols::IsMap => builder.ins().is_type(Type::Term(TermType::Map), arg, span),
                    symbols::IsNumber => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Number), arg, span)
                    }
                    symbols::IsPid => builder.ins().is_type(Type::Term(TermType::Pid), arg, span),
                    symbols::IsPort => builder.ins().is_type(Type::Term(TermType::Port), arg, span),
                    symbols::IsReference => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Reference), arg, span)
                    }
                    symbols::IsTuple => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Tuple(None)), arg, span)
                    }
                    symbols::IsFunction if op.arity == 1 => {
                        builder
                            .ins()
                            .is_type(Type::Term(TermType::Fun(None)), arg, span)
                    }
                    symbols::IsRecord => panic!("expected is_record to be handled in lower_bif"),
                    _ => unimplemented!("unsupported type test {}", &op),
                };
                builder.ins().br_unless(result, fail, &[], span);
                Ok(())
            }
            _ if op.is_comparison_op() => {
                let args = self.ssa_values(builder, args)?;
                let result = match op.function {
                    symbols::Equal => builder.ins().eq(args[0], args[1], span),
                    symbols::NotEqual => builder.ins().neq(args[0], args[1], span),
                    symbols::EqualStrict => builder.ins().eq_exact(args[0], args[1], span),
                    symbols::NotEqualStrict => builder.ins().neq_exact(args[0], args[1], span),
                    symbols::Gte => builder.ins().gte(args[0], args[1], span),
                    symbols::Gt => builder.ins().gt(args[0], args[1], span),
                    symbols::Lte => builder.ins().lte(args[0], args[1], span),
                    symbols::Lt => builder.ins().lt(args[0], args[1], span),
                    _ => unimplemented!("unsupported comparison test {}", op),
                };
                builder.ins().br_unless(result, fail, &[], span);
                Ok(())
            }
            _ => {
                let callee = self.module.get_or_register_builtin(op);
                let args = self.ssa_values(builder, args)?;
                let inst = builder.ins().call(callee, args.as_slice(), span);
                let result = builder.first_result(inst);
                builder.ins().br_unless(result, fail, &[], span);
                Ok(())
            }
        }
    }

    fn lower_test_is_record<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        tuple: Value,
        tag: Symbol,
        arity: usize,
        fail: Block,
    ) -> anyhow::Result<()> {
        // First, check that the given value is a tuple and of the correct arity
        let ty = Type::tuple(arity);
        let is_type = builder.ins().is_type(ty.clone(), tuple, span);
        builder.ins().br_unless(is_type, fail, &[], span);
        // Cast the input to the correct tuple type now that we know it is the expected shape
        let tuple = builder.ins().cast(tuple, ty, span);
        // Fetch the tag element of the tuple
        let elem = builder.ins().get_element_imm(tuple, 0, span);
        // Compare the fetched tag to the expected tag, branching to the fail block if there is a
        // mismatch
        let tag = builder.ins().atom(tag, span);
        let has_tag = builder.ins().eq_exact(elem, tag, span);
        builder.ins().br_unless(has_tag, fail, &[], span);
        Ok(())
    }

    fn lower_call<'a>(&mut self, builder: &'a mut IrBuilder, call: k::Call) -> anyhow::Result<()> {
        let span = call.span();
        match self.fail_context() {
            // Inside a guard. The only allowed function call is to erlang:error/1,2.
            // We will generate a branch to the failure branch.
            FailContext::Guard(_) => panic!(
                "invalid callee in guard {:#?}/{}",
                call.callee.as_ref(),
                call.args.len()
            ),
            // Ordinary function call in a function body.
            _fail => {
                let inst = match *call.callee {
                    KExpr::Bif(k::Bif {
                        span: bif_span,
                        op,
                        args: mut env,
                        ..
                    }) => {
                        // Call to a local closure
                        assert_eq!(op.module, Some(symbols::Erlang));
                        assert_eq!(op.function, symbols::MakeFun);
                        // NOTE: This is an optimization, because we know that we
                        // are calling a closure, and that the closure is in the local
                        // module, we can skip the overhead of erlang:apply/2 and skip
                        // straight to calling the callee. However, we must still construct
                        // the closure, as it contains the env that the target function expects
                        // as an extra argument
                        let KExpr::Local(name) = env.remove(0) else { panic!("expected local here") };
                        let env = self.ssa_values(builder, env)?;
                        let func = builder
                            .get_callee(name.item)
                            .expect("undefined local function reference");
                        let make_fun = builder.ins().make_fun(func, env.as_slice(), bif_span);
                        let fun = builder.first_result(make_fun);
                        let mut args = self.ssa_values(builder, call.args)?;
                        args.push(fun);
                        builder.ins().call(func, args.as_slice(), span)
                    }
                    KExpr::Local(name) | KExpr::Remote(k::Remote::Static(name)) => {
                        assert!(
                            !self.module.is_closure(&name),
                            "expected static calls to closures to be transformed"
                        );
                        // Static call to a regular function
                        let args = self.ssa_values(builder, call.args)?;
                        let func = builder.get_or_register_callee(name.item);
                        builder.ins().call(func, args.as_slice(), span)
                    }
                    KExpr::Remote(k::Remote::Dynamic(module, function)) => {
                        // Indirect callee to a full MFA, convert to an erlang:apply/3 call
                        let module = self.ssa_value(builder, *module)?;
                        let function = self.ssa_value(builder, *function)?;
                        let mut args = self.ssa_values(builder, call.args)?;
                        let apply3 = FunctionName::new(symbols::Erlang, symbols::Apply, 3);
                        let apply3 = self.module.get_or_register_builtin(apply3);
                        let argv = args.drain(..).rfold(builder.ins().nil(span), |tail, hd| {
                            builder.ins().cons(hd, tail, span)
                        });
                        builder.ins().call(apply3, &[module, function, argv], span)
                    }
                    v @ KExpr::Var(_) => {
                        let callee = self.ssa_value(builder, v)?;
                        let args = self.ssa_values(builder, call.args)?;
                        // The callee is known statically to be a fun, so we can use the optimized
                        // call path
                        builder
                            .ins()
                            .call_indirect(callee, CallConv::Erlang, args.as_slice(), span)
                    }
                    other => panic!("unexpected callee expression: {:#?}", &other),
                };
                let result = builder.first_result(inst);
                if let Some(ret) = call.ret.first().map(|e| e.as_var().unwrap().name()) {
                    builder.define_var(ret, result);
                }
                //let landing_pad = fail.block();
                //builder.ins().br_if(is_err, landing_pad, &[result], span);
                Ok(())
            }
        }
    }

    fn lower_enter<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        call: k::Enter,
    ) -> anyhow::Result<()> {
        assert_matches!(self.fail_context(), FailContext::Uncaught(_));

        // Ordinary function call in a function body.
        let span = call.span();
        match *call.callee {
            KExpr::Bif(k::Bif {
                span: bif_span,
                op,
                args: mut env,
                ..
            }) => {
                // Call to a local closure
                assert_eq!(op.module, Some(symbols::Erlang));
                assert_eq!(op.function, symbols::MakeFun);
                // NOTE: This is an optimization, because we know that we
                // are calling a closure, and that the closure is in the local
                // module, we can skip the overhead of erlang:apply/2 and skip
                // straight to calling the callee. However, we must still construct
                // the closure, as it contains the env that the target function expects
                // as an extra argument
                let KExpr::Local(name) = env.remove(0) else { panic!("expected local here") };
                let env = self.ssa_values(builder, env)?;
                let func = builder
                    .get_callee(name.item)
                    .expect("undefined local function reference");
                let make_fun = builder.ins().make_fun(func, env.as_slice(), bif_span);
                let fun = builder.first_result(make_fun);
                // Lastly, call the closure function directly
                let mut args = self.ssa_values(builder, call.args)?;
                args.push(fun);
                builder.ins().enter(func, args.as_slice(), span)
            }
            KExpr::Local(name) | KExpr::Remote(k::Remote::Static(name)) => {
                // Static call to a regular function
                assert!(
                    !self.module.is_closure(&name),
                    "expected static calls to closures to be transformed"
                );
                let args = self.ssa_values(builder, call.args)?;
                let func = builder.get_or_register_callee(name.item);
                builder.ins().enter(func, args.as_slice(), span)
            }
            KExpr::Remote(k::Remote::Dynamic(module, function)) => {
                // Indirect callee to a full MFA, convert to an erlang:apply/3 call
                let module = self.ssa_value(builder, *module)?;
                let function = self.ssa_value(builder, *function)?;
                let mut args = self.ssa_values(builder, call.args)?;
                let apply3 = FunctionName::new(symbols::Erlang, symbols::Apply, 3);
                let apply3 = self.module.get_or_register_builtin(apply3);
                let argv = args.drain(..).rfold(builder.ins().nil(span), |tail, hd| {
                    builder.ins().cons(hd, tail, span)
                });
                builder.ins().enter(apply3, &[module, function, argv], span)
            }
            v @ KExpr::Var(_) => {
                // Indirect callee to a fun
                let is_closure = v.has_annotation(symbols::Closure);
                let callee = self.ssa_value(builder, v)?;
                // Optimize the case where we know that the callee is a fun that we just created
                let mut args = self.ssa_values(builder, call.args)?;
                if is_closure {
                    // The callee is known statically to be a fun, so we can use the optimized call
                    // path
                    builder
                        .ins()
                        .enter_indirect(callee, CallConv::Erlang, args.as_slice(), span)
                } else {
                    // The callee is either not a fun at all, or we are unable to verify, use the
                    // safe path by converting this to a call to apply/2
                    let apply2 = FunctionName::new(symbols::Erlang, symbols::Apply, 2);
                    let apply2 = self.module.get_or_register_builtin(apply2);
                    let argv = args.drain(..).rfold(builder.ins().nil(span), |tail, hd| {
                        builder.ins().cons(hd, tail, span)
                    });
                    builder.ins().enter(apply2, &[callee, argv], span)
                }
            }
            other => panic!("unexpected callee expression: {:#?}", &other),
        };
        Ok(())
    }

    ///  Generate code for a guard BIF or primop.
    fn lower_bif<'a>(&mut self, builder: &'a mut IrBuilder, bif: k::Bif) -> anyhow::Result<()> {
        let span = bif.span();
        assert_eq!(bif.op.module, Some(symbols::Erlang));
        if bif.op.is_primop() {
            return self.lower_internal(builder, bif);
        }
        if bif.op.is_operator() {
            return self.lower_operator(builder, bif);
        }
        match (bif.op.function, bif.args.as_slice()) {
            (
                symbols::IsRecord,
                [_, KExpr::Literal(Literal {
                    value: Lit::Atom(tag),
                    ..
                }), KExpr::Literal(Literal {
                    value: Lit::Integer(arity),
                    ..
                })],
            ) => {
                let tag = *tag;
                let arity = arity.to_usize().unwrap();
                self.lower_is_record_bif(builder, bif, tag, arity)
            }
            _ if bif.op.is_type_test() => self.lower_type_test(builder, bif),
            _ if bif.op.is_safe() => {
                // This bif can never fail, and has no side effects
                let callee = self.module.get_or_register_builtin(bif.op);
                let args = self.ssa_values(builder, bif.args)?;
                let inst = builder.ins().call(callee, args.as_slice(), span);
                let mut results = builder.inst_results(inst).to_vec();
                assert_eq!(
                    bif.ret.len(),
                    results.len(),
                    "expected bif {} to have {} results",
                    bif.op,
                    results.len(),
                );
                for (ret, value) in bif
                    .ret
                    .iter()
                    .map(|e| e.as_var().map(|v| v.name()).unwrap())
                    .zip(results.drain(..))
                {
                    builder.define_var(ret, value);
                }
                Ok(())
            }
            _ => {
                // This bif is fallible, and may have side effects, so must be treated like a
                // standard call
                assert!(bif.ret.len() <= 1);
                let callee = self.module.get_or_register_builtin(bif.op);
                let args = self.ssa_values(builder, bif.args)?;
                let inst = builder.ins().call(callee, args.as_slice(), span);
                let result = builder.first_result(inst);
                // If there are no rets, the callee must raise an exception
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), result);
                }
                Ok(())
            }
        }
    }

    fn lower_is_record_bif<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut bif: k::Bif,
        tag: Symbol,
        arity: usize,
    ) -> anyhow::Result<()> {
        let span = bif.span();
        let tuple = self.ssa_value(builder, bif.args.remove(0))?;

        // Construct a flow control structure that goes something like this:
        //
        //     $0 = is_type($tuple, arity)
        //     if $0 {
        //         $tag = get_element($tuple, 0)
        //         $1 = eq.strict $tag, $0
        //     } else {
        //         $1 = false
        //     }
        //     ...
        let tuple_type = Type::tuple(arity);
        let is_type = builder.ins().is_type(tuple_type.clone(), tuple, span);
        let tag_check_block = builder.create_block();
        let final_block = builder.create_block();
        builder.append_block_param(final_block, Type::Term(TermType::Bool), span);
        builder.ins().br_if(is_type, tag_check_block, &[], span);
        builder.ins().br(final_block, &[is_type], span);
        builder.switch_to_block(tag_check_block);
        let tag_value = builder.ins().get_element_imm(tuple, 0, span);
        let has_tag = builder.ins().eq_exact_imm(tag_value, tag.into(), span);
        builder.ins().br(final_block, &[has_tag], span);
        builder.switch_to_block(final_block);
        Ok(())
    }

    fn lower_type_test<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        bif: k::Bif,
    ) -> anyhow::Result<()> {
        let span = bif.span();
        debug_assert_eq!(bif.op.module, Some(symbols::Erlang));
        let args = self.ssa_values(builder, bif.args)?;
        assert_eq!(args.len(), bif.op.arity as usize);
        let result = match bif.op.function {
            symbols::IsAtom => builder
                .ins()
                .is_type(Type::Term(TermType::Atom), args[0], span),
            symbols::IsBinary => builder
                .ins()
                .is_type(Type::Term(TermType::Binary), args[0], span),
            symbols::IsBitstring => {
                builder
                    .ins()
                    .is_type(Type::Term(TermType::Bitstring), args[0], span)
            }
            symbols::IsBoolean => builder
                .ins()
                .is_type(Type::Term(TermType::Bool), args[0], span),
            symbols::IsFloat => builder
                .ins()
                .is_type(Type::Term(TermType::Float), args[0], span),
            symbols::IsInteger => {
                builder
                    .ins()
                    .is_type(Type::Term(TermType::Integer), args[0], span)
            }
            symbols::IsList => {
                builder
                    .ins()
                    .is_type(Type::Term(TermType::List(None)), args[0], span)
            }
            symbols::IsMap => builder
                .ins()
                .is_type(Type::Term(TermType::Map), args[0], span),
            symbols::IsNumber => builder
                .ins()
                .is_type(Type::Term(TermType::Number), args[0], span),
            symbols::IsPid => builder
                .ins()
                .is_type(Type::Term(TermType::Pid), args[0], span),
            symbols::IsPort => builder
                .ins()
                .is_type(Type::Term(TermType::Port), args[0], span),
            symbols::IsReference => {
                builder
                    .ins()
                    .is_type(Type::Term(TermType::Reference), args[0], span)
            }
            symbols::IsTuple => {
                builder
                    .ins()
                    .is_type(Type::Term(TermType::Tuple(None)), args[0], span)
            }
            symbols::IsFunction => match bif.op.arity {
                1 => builder
                    .ins()
                    .is_type(Type::Term(TermType::Fun(None)), args[0], span),
                2 => builder.ins().is_function_with_arity(args[0], args[1], span),
                _ => unimplemented!("invalid type types bif: {}", &bif.op),
            },
            symbols::IsRecord => panic!("expected is_record to be handled in lower_bif"),
            _ => unimplemented!("unsupported type test {}", &bif.op),
        };

        assert_eq!(bif.ret.len(), 1);
        builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), result);
        Ok(())
    }

    fn lower_operator<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        bif: k::Bif,
    ) -> anyhow::Result<()> {
        let span = bif.span();
        debug_assert_eq!(bif.op.module, Some(symbols::Erlang));
        let args = self.ssa_values(builder, bif.args)?;
        assert_eq!(args.len(), bif.op.arity as usize);
        let result = match bif.op.function {
            symbols::Plus => builder.ins().add(args[0], args[1], span),
            symbols::Minus if bif.op.arity == 1 => builder.ins().neg(args[0], span),
            symbols::Minus => builder.ins().sub(args[0], args[1], span),
            symbols::Bnot => builder.ins().bnot(args[0], span),
            symbols::Star => builder.ins().mul(args[0], args[1], span),
            symbols::Slash => builder.ins().fdiv(args[0], args[1], span),
            symbols::Div => builder.ins().div(args[0], args[1], span),
            symbols::Rem => builder.ins().rem(args[0], args[1], span),
            symbols::Band => builder.ins().band(args[0], args[1], span),
            symbols::Bor => builder.ins().bor(args[0], args[1], span),
            symbols::Bxor => builder.ins().bxor(args[0], args[1], span),
            symbols::Bsl => builder.ins().bsl(args[0], args[1], span),
            symbols::Bsr => builder.ins().bsr(args[0], args[1], span),
            symbols::Not => builder.ins().not(args[0], span),
            symbols::And => builder.ins().and(args[0], args[1], span),
            symbols::Or => builder.ins().or(args[0], args[1], span),
            symbols::Xor => builder.ins().xor(args[0], args[1], span),
            symbols::PlusPlus => builder.ins().list_concat(args[0], args[1], span),
            symbols::MinusMinus => builder.ins().list_subtract(args[0], args[1], span),
            symbols::Bang => builder.ins().send(args[0], args[1], span),
            symbols::Equal => builder.ins().eq(args[0], args[1], span),
            symbols::NotEqual => builder.ins().neq(args[0], args[1], span),
            symbols::EqualStrict => builder.ins().eq_exact(args[0], args[1], span),
            symbols::NotEqualStrict => builder.ins().neq_exact(args[0], args[1], span),
            symbols::Gte => builder.ins().gte(args[0], args[1], span),
            symbols::Gt => builder.ins().gt(args[0], args[1], span),
            symbols::Lte => builder.ins().lte(args[0], args[1], span),
            symbols::Lt => builder.ins().lt(args[0], args[1], span),
            symbols::Hd => builder.ins().head(args[0], span),
            symbols::Tl => builder.ins().tail(args[0], span),
            _ => unimplemented!("unsupported type test {}", &bif.op),
        };

        assert_eq!(bif.ret.len(), 1);
        builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), result);
        Ok(())
    }

    fn lower_internal<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut bif: k::Bif,
    ) -> anyhow::Result<()> {
        let span = bif.span();
        match (bif.op.function, bif.args.as_slice()) {
            (symbols::GarbageCollect, _) => {
                assert_eq!(bif.args.len(), 0);
                assert!(bif.ret.len() <= 1);
                let success = builder.ins().garbage_collect(span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), success);
                }
            }
            (symbols::MakeFun, [KExpr::Local(local), ..]) => {
                // make_fun/2 requires special handling to convert to its corresponding core
                // instruction
                let callee = builder
                    .get_callee(local.item)
                    .expect("undefined local function reference");
                let callee_type = builder.func.dfg.callee_signature(callee).get_type().clone();
                let env = self.ssa_values(builder, bif.args.split_off(1))?;
                let inst = builder.ins().make_fun(callee, env.as_slice(), span);
                let fun = builder.first_result(inst);
                if !bif.ret.is_empty() {
                    let var = bif.ret[0].as_var().map(|v| v.name()).unwrap();
                    builder.define_var(var, fun);
                    builder
                        .set_var_type(var, Type::Term(TermType::Fun(Some(Box::new(callee_type)))));
                }
            }
            (symbols::MakeFun, [KExpr::Remote(k::Remote::Static(name)), ..]) => {
                let callee = builder.get_or_register_callee(name.item);
                let callee_type = builder.func.dfg.callee_signature(callee).get_type().clone();
                let env = self.ssa_values(builder, bif.args.split_off(1))?;
                let inst = builder.ins().make_fun(callee, env.as_slice(), span);
                let fun = builder.first_result(inst);
                if !bif.ret.is_empty() {
                    let var = bif.ret[0].as_var().map(|v| v.name()).unwrap();
                    builder.define_var(var, fun);
                    builder
                        .set_var_type(var, Type::Term(TermType::Fun(Some(Box::new(callee_type)))));
                }
            }
            (symbols::MakeFun, _) => panic!("unexpected make_fun bif arguments: {:#?}", &bif.args),
            (symbols::UnpackEnv, _) => {
                assert_eq!(
                    bif.args.len(),
                    2,
                    "expected unpack_env bif to have two arguments"
                );
                assert_eq!(bif.ret.len(), 1, "result of unpack_env bif must be used");
                let index = match bif.args.pop().unwrap() {
                    KExpr::Literal(Literal { value: Lit::Integer(Int::Small(i)), .. }) => i,
                    other => panic!("invalid argument given to unpack_env bif, expected integer literal, got: {:#?}", &other),
                };
                let fun = self.ssa_value(builder, bif.args.pop().unwrap())?;
                let value =
                    builder
                        .ins()
                        .unpack_env(fun, index.try_into().expect("index too large"), span);
                builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), value);
            }
            (symbols::RemoveMessage, _) => {
                assert_eq!(bif.ret.len(), 0);
                assert_eq!(bif.args.len(), 0);
                builder.ins().recv_pop(span);
            }
            (symbols::RecvNext, _) => {
                assert_eq!(bif.ret.len(), 0);
                assert_eq!(bif.args.len(), 0);
                builder.ins().recv_next(span);
            }
            (symbols::RecvPeekMessage, _) => {
                let (available, message) = builder.ins().recv_peek(span);
                builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), available);
                builder.define_var(bif.ret[1].as_var().map(|v| v.name()).unwrap(), message);
            }
            (symbols::RecvWaitTimeout, _) => {
                assert_eq!(bif.args.len(), 1);
                assert_eq!(bif.ret.len(), 1);
                let timeout = self.ssa_value(builder, bif.args.pop().unwrap())?;
                let timed_out = builder.ins().recv_wait_timeout(timeout, span);
                builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), timed_out);
            }
            (symbols::BuildStacktrace, _) => {
                assert_eq!(
                    bif.args.len(),
                    1,
                    "invalid number of arguments for build_stacktrace bif"
                );
                assert_eq!(
                    bif.ret.len(),
                    1,
                    "result of build_stacktrace bif must be used"
                );
                let raw_stk = self.ssa_value(builder, bif.args.pop().unwrap())?;
                let trace = builder.ins().stacktrace(raw_stk, span);
                builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), trace);
            }
            // The nif_start instruction is simply a marker for now, we don't have any reason to
            // emit it to SSA
            (symbols::NifStart, _) => {
                assert_eq!(
                    bif.args.len(),
                    0,
                    "invalid number of arguments for nif_start bif"
                );
                assert_eq!(
                    bif.ret.len(),
                    0,
                    "nif_start bif does not produce results, but some are expected"
                );
            }
            // MatchFail is a special exception builtin that requires some extra treatment
            (symbols::MatchFail, _) => {
                // If this is a function or case clause error, the arity is dynamic, but we need
                // to convert the argument list into an appropriate form for calling
                // erlang:match_fail/2
                match bif.args[0].as_atom() {
                    Some(symbols::FunctionClause) => {
                        let mut args = self.ssa_values(builder, bif.args)?;
                        let ty = args.remove(0);
                        let argv = args.drain(..).rfold(builder.ins().nil(span), |tail, head| {
                            builder.ins().cons(head, tail, span)
                        });
                        // The first argument will be the type of match error (function),
                        // the second will be a tuple containing the name of the module,
                        // the name of the function, and a list of the arguments, and optionally,
                        // a list of extra info (e.g. file/line)
                        let (module, function) = match bif.annotations.get(symbols::Inlined) {
                            None => {
                                // This error is for the current module/function
                                let module = builder.ins().atom(self.signature.module, span);
                                let function = builder.ins().atom(self.signature.name, span);
                                (module, function)
                            }
                            Some(Annotation::Term(Literal {
                                value: Lit::Tuple(elements),
                                ..
                            })) if elements.len() == 2 => {
                                let Literal { value: Lit::Atom(name), .. } = elements[0] else { panic!("expected literal atom, got: {:#?}", &elements[0]) };
                                // This error was inlined from another function which we can
                                // extract from the annotated {Name, Arity} tuple
                                let module = builder.ins().atom(self.signature.module, span);
                                let function = builder.ins().atom(name, span);
                                (module, function)
                            }
                            other => panic!("unexpected inlined attribute value: {:#?}", &other),
                        };
                        let meta = builder.ins().nil(span);
                        let reason = builder.ins().tuple_imm(4, span);
                        builder.ins().set_element_mut(reason, 0, module, span);
                        builder.ins().set_element_mut(reason, 1, function, span);
                        builder.ins().set_element_mut(reason, 2, argv, span);
                        builder.ins().set_element_mut(reason, 3, meta, span);
                        let error = builder.ins().tuple_imm(2, span);
                        builder.ins().set_element_mut(error, 0, ty, span);
                        builder.ins().set_element_mut(error, 1, reason, span);
                        builder.ins().error(error, span);
                        if !bif.ret.is_empty() {
                            builder
                                .define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), error);
                        }
                    }
                    Some(symbols::CaseClause) => {
                        // The first argument will be the type of match error (case clause),
                        // the second will be a list of the arguments
                        let mut args = self.ssa_values(builder, bif.args)?;
                        let ty = args.remove(0);
                        let argv = args.drain(..).rfold(builder.ins().nil(span), |tail, head| {
                            builder.ins().cons(head, tail, span)
                        });
                        let error = builder.ins().tuple_imm(2, span);
                        builder.ins().set_element_mut(error, 0, ty, span);
                        builder.ins().set_element_mut(error, 1, argv, span);
                        builder.ins().error(error, span);
                        if !bif.ret.is_empty() {
                            builder
                                .define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), error);
                        }
                    }
                    _ => {
                        // This is a regular match error, in which there is a single argument
                        assert_eq!(bif.args.len(), 2);
                        let reason = self.ssa_value(builder, bif.args.pop().unwrap())?;
                        let ty = self.ssa_value(builder, bif.args.pop().unwrap())?;
                        let error = builder.ins().tuple_imm(2, span);
                        builder.ins().set_element_mut(error, 0, ty, span);
                        builder.ins().set_element_mut(error, 1, reason, span);
                        builder.ins().error(error, span);
                        if !bif.ret.is_empty() {
                            builder
                                .define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), error);
                        }
                    }
                }
            }
            (symbols::Halt, _) => {
                assert!(bif.ret.len() <= 1);
                let args = self.ssa_values(builder, bif.args).unwrap();
                builder.ins().halt(&args, span);
                if !bif.ret.is_empty() {
                    let reason = builder.ins().atom(symbols::Halt, span);
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), reason);
                }
            }
            (symbols::Throw, _) => {
                assert_eq!(bif.args.len(), 1);
                assert!(bif.ret.len() <= 1);
                let reason = self.ssa_value(builder, bif.args.pop().unwrap())?;
                builder.ins().throw(reason, span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), reason);
                }
            }
            (symbols::Error, _) => {
                assert!(!bif.args.is_empty());
                assert!(bif.ret.len() <= 1);
                let args = self.ssa_values(builder, bif.args).unwrap();
                let reason = args[0];
                match args.len() {
                    1 => {
                        builder.ins().error(reason, span);
                    }
                    2 => {
                        let kind = builder.ins().atom(symbols::Error, span);
                        builder.ins().raise(kind, reason, args[1], span);
                    }
                    3 => {
                        let kind = builder.ins().atom(symbols::Error, span);
                        builder
                            .ins()
                            .raise_with_opts(kind, reason, args[1], args[2], span);
                    }
                    _ => panic!("invalid error bif: {}", &bif.op),
                }
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), reason);
                }
            }
            (symbols::Exit, args) if args.len() == 1 => {
                assert!(bif.ret.len() <= 1);
                let reason = self.ssa_value(builder, bif.args.pop().unwrap())?;
                builder.ins().exit1(reason, span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), reason);
                }
            }
            (symbols::Exit, _) => {
                assert_eq!(bif.args.len(), 2);
                assert!(bif.ret.len() <= 1);
                let args = self.ssa_values(builder, bif.args)?;
                let result = builder.ins().exit2(args[0], args[1], span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), result);
                }
            }
            (symbols::NifError, _) => {
                assert_eq!(bif.args.len(), 1);
                assert!(bif.ret.len() <= 1);
                let reason = self.ssa_value(builder, bif.args.pop().unwrap())?;
                builder.ins().error(reason, span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), reason);
                }
            }
            (symbols::Raise, _) => {
                assert_eq!(bif.args.len(), 3);
                assert!(bif.ret.len() <= 1);
                let args = self.ssa_values(builder, bif.args).unwrap();
                let badarg = builder.ins().raise(args[0], args[1], args[2], span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), badarg);
                }
            }
            (op, _) if bif.op.is_exception_op() => unimplemented!("{:?}", op),
            (symbols::Yield, _) => {
                assert_eq!(bif.args.len(), 0);
                assert!(bif.ret.len() <= 1);
                let yielded = builder.ins().r#yield(span);
                if !bif.ret.is_empty() {
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), yielded);
                }
            }
            _ => {
                let callee = self.module.get_or_register_builtin(bif.op);
                // All other primops behave like regular function calls
                let args = self.ssa_values(builder, bif.args)?;
                let inst = builder.ins().call(callee, args.as_slice(), span);
                let result = builder.first_result(inst);
                if !bif.ret.is_empty() {
                    assert_eq!(
                        bif.ret.len(),
                        1,
                        "mismatch in the number of expected results for builtin {}",
                        bif.op
                    );
                    builder.define_var(bif.ret[0].as_var().map(|v| v.name()).unwrap(), result);
                }
            }
        }
        Ok(())
    }

    fn lower_try<'a>(&mut self, builder: &'a mut IrBuilder, expr: k::Try) -> anyhow::Result<()> {
        let span = expr.span();
        let current_block = builder.current_block();

        let body_block = builder.create_block();
        for var in expr.vars.iter() {
            let value =
                builder.append_block_param(body_block, Type::Term(TermType::Any), var.span());
            builder.define_var(var.name(), value);
        }

        let handler_block = builder.create_block();
        let kind = builder.append_block_param(handler_block, Type::Term(TermType::Atom), span);
        let reason = builder.append_block_param(handler_block, Type::Term(TermType::Any), span);
        let trace = builder.append_block_param(handler_block, Type::Exception, span);
        builder.switch_to_block(handler_block);
        for (evar, value) in expr
            .evars
            .iter()
            .map(|v| v.name())
            .zip(&[kind, reason, trace])
        {
            builder.define_var(evar, *value);
        }

        let final_block = builder.create_block();
        for var in expr.ret.iter().map(|e| e.as_var().unwrap()) {
            let value =
                builder.append_block_param(final_block, Type::Term(TermType::Any), var.span());
            builder.define_var(var.name(), value);
        }

        builder.switch_to_block(current_block);
        self.brk.push(body_block);
        self.landing_pads.push(handler_block);
        builder.ins().start_catch(handler_block, span);
        self.lower(builder, *expr.arg)?;
        self.brk.pop();
        self.landing_pads.pop();

        builder.switch_to_block(body_block);
        builder.ins().end_catch(span);
        self.brk.push(final_block);
        self.lower(builder, *expr.body)?;

        builder.switch_to_block(handler_block);
        builder.ins().end_catch(span);
        self.lower(builder, *expr.handler)?;
        self.brk.pop();

        builder.switch_to_block(final_block);

        Ok(())
    }

    fn lower_try_enter<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: k::TryEnter,
    ) -> anyhow::Result<()> {
        let span = expr.span();
        let current_block = builder.current_block();

        let body_block = builder.create_block();
        for var in expr.vars.iter() {
            let value =
                builder.append_block_param(body_block, Type::Term(TermType::Any), var.span());
            builder.define_var(var.name(), value);
        }

        let handler_block = builder.create_block();
        let kind = builder.append_block_param(handler_block, Type::Term(TermType::Atom), span);
        let reason = builder.append_block_param(handler_block, Type::Term(TermType::Any), span);
        let trace = builder.append_block_param(handler_block, Type::Term(TermType::Any), span);
        builder.switch_to_block(handler_block);
        for (evar, value) in expr
            .evars
            .iter()
            .map(|v| v.name())
            .zip(&[kind, reason, trace])
        {
            builder.define_var(evar, *value);
        }

        builder.switch_to_block(current_block);
        self.brk.push(body_block);
        self.landing_pads.push(handler_block);
        builder.ins().start_catch(handler_block, span);
        self.lower(builder, *expr.arg)?;
        self.brk.pop();
        self.landing_pads.pop();

        builder.switch_to_block(body_block);
        builder.ins().end_catch(span);
        self.lower(builder, *expr.body)?;

        builder.switch_to_block(handler_block);
        builder.ins().end_catch(span);
        self.lower(builder, *expr.handler)?;
        self.landing_pads.pop();

        Ok(())
    }

    fn lower_catch<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: k::Catch,
    ) -> anyhow::Result<()> {
        assert_eq!(expr.ret.len(), 1);
        let ret = expr.ret[0].as_var().map(|v| v.name()).unwrap();
        let span = expr.span();
        let current_block = builder.current_block();

        let handler_block = builder.create_block();
        let kind = builder.append_block_param(handler_block, Type::Term(TermType::Atom), span);
        let reason = builder.append_block_param(handler_block, Type::Term(TermType::Any), span);
        let trace = builder.append_block_param(handler_block, Type::Term(TermType::Any), span);

        // The result block is where the fork in control is rejoined, it receives a single block
        // argument which is either the normal return value, or the caught/wrapped exception
        // value
        let result_block = builder.create_block();
        let result = builder.append_block_param(result_block, Type::Term(TermType::Any), span);
        builder.define_var(ret, result);

        // The exit block handles wrapping exit/error reasons in the {'EXIT', Reason} tuple
        // It receives a single block argument which corresponds to `Reason` in the previous
        // sentence.
        let exit_block = builder.create_block();
        let exit_reason = builder.append_block_param(exit_block, Type::Term(TermType::Any), span);

        builder.switch_to_block(handler_block);
        // Throws are the most common, and require no special handling, so we jump straight to the
        // result block for them
        let is_throw = builder
            .ins()
            .eq_exact_imm(kind, symbols::Throw.into(), span);
        builder.ins().br_if(is_throw, result_block, &[reason], span);
        // Exits are the next simplest, as we just wrap the reason in a tuple, so we jump straight
        // to the exit block
        let is_exit = builder.ins().eq_exact_imm(kind, symbols::Exit.into(), span);
        builder.ins().br_if(is_exit, exit_block, &[reason], span);
        // We have to construct a new error reason, and then jump to the exit block to wrap it in
        // the exit tuple
        let error_reason = builder.ins().tuple_imm(2, span);
        let error_reason = builder.ins().set_element_mut(error_reason, 0, reason, span);
        let error_reason = builder.ins().set_element_mut(error_reason, 1, trace, span);
        builder.ins().br(exit_block, &[error_reason], span);

        // In the exit block, we need just to construct the {'EXIT', Reason} tuple, and then jump to
        // the result block
        builder.switch_to_block(exit_block);
        let wrapped_reason = builder.ins().tuple_imm(2, span);
        let wrapped_reason =
            builder
                .ins()
                .set_element_mut_imm(wrapped_reason, 0, symbols::EXIT.into(), span);
        let wrapped_reason = builder
            .ins()
            .set_element_mut(wrapped_reason, 1, exit_reason, span);
        builder.ins().br(result_block, &[wrapped_reason], span);

        // Lower body
        builder.switch_to_block(current_block);
        self.brk.push(result_block);
        self.landing_pads.push(handler_block);
        builder.ins().start_catch(handler_block, span);
        self.lower(builder, *expr.body)?;
        self.brk.pop();
        self.landing_pads.pop();

        builder.switch_to_block(result_block);
        builder.ins().end_catch(span);

        Ok(())
    }

    fn lower_put<'a>(&mut self, builder: &'a mut IrBuilder, expr: k::Put) -> anyhow::Result<()> {
        let span = expr.span();
        let ret = expr.ret[0].as_var().map(|v| v.name()).unwrap();
        match *expr.arg {
            KExpr::Cons(k::Cons {
                box head, box tail, ..
            }) => {
                let head = self.ssa_value(builder, head)?;
                let tail = self.ssa_value(builder, tail)?;
                let list = builder.ins().cons(head, tail, span);
                builder.define_var(ret, list);
                Ok(())
            }
            KExpr::Tuple(k::Tuple { span, elements, .. }) => {
                let mut elements = self.ssa_values(builder, elements)?;
                let tuple = builder.ins().tuple_imm(elements.len(), span);
                for (i, element) in elements.drain(..).enumerate() {
                    builder.ins().set_element_mut(tuple, i, element, span);
                }
                builder.define_var(ret, tuple);
                Ok(())
            }
            KExpr::Binary(k::Binary { box segment, .. }) => {
                self.lower_binary(builder, span, ret, segment)
            }
            KExpr::Map(k::Map {
                span,
                op,
                box var,
                mut pairs,
                ..
            }) if pairs.len() == 1 && pairs[0].is_var_key() => {
                // Single variable key
                let map = self.ssa_value(builder, var)?;
                let pair = pairs.pop().unwrap();
                let k = self.ssa_value(builder, *pair.key)?;
                let v = self.ssa_value(builder, *pair.value)?;
                self.lower_map(builder, span, ret, op, map, vec![(k, v)])
            }
            KExpr::Map(k::Map {
                span,
                op,
                box var,
                mut pairs,
                ..
            }) => {
                // One or more literal keys
                assert!(
                    pairs.iter().all(|pair| !pair.is_var_key()),
                    "expected only literal keys"
                );
                let map = self.ssa_value(builder, var)?;
                let kvs = pairs
                    .drain(..)
                    .map(|pair| {
                        let k = self.ssa_value(builder, *pair.key)?;
                        let v = self.ssa_value(builder, *pair.value)?;
                        Ok::<_, anyhow::Error>((k, v))
                    })
                    .try_collect()?;
                self.lower_map(builder, span, ret, op, map, kvs)
            }
            expr => {
                // Create an alias for a variable or literal
                let value = self.ssa_value(builder, expr)?;
                builder.define_var(ret, value);
                Ok(())
            }
        }
    }

    fn lower_map<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        ret: Symbol,
        op: MapOp,
        map: Value,
        mut pairs: Vec<(Value, Value)>,
    ) -> anyhow::Result<()> {
        // We know that all insertions except the first appear atomic,
        // so we can optimize the inserts by only doing an immutable insert
        // on the first pair, and mutably inserting the remaining pairs
        let kv = pairs
            .drain(..)
            .map(|(k, v)| [k, v])
            .flatten()
            .collect::<SmallVec<[Value; 8]>>();
        let map = match op {
            MapOp::Assoc => builder.ins().map_extend_put(map, &kv, span),
            MapOp::Exact => builder.ins().map_extend_update(map, &kv, span),
        };
        builder.define_var(ret, map);
        Ok(())
    }

    fn lower_binary<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        ret: Symbol,
        mut segment: k::Expr,
    ) -> anyhow::Result<()> {
        // TODO: We should create an equivalent to bs_create_bin that allows us to
        // calculate the runtime size of the constructed binary and do validation
        // all in one mega-instruction since it allows for optimization opportunities
        // that this flow does not
        let mut bin = builder.ins().bs_init(span);
        loop {
            match segment {
                KExpr::BinarySegment(seg) => {
                    let spec = seg.spec;
                    let value = self.ssa_value(builder, *seg.value)?;
                    let size = match seg.size {
                        None
                        | Some(box KExpr::Literal(Literal {
                            value: Lit::Atom(symbols::All),
                            ..
                        })) => None,
                        Some(box expr) => Some(self.ssa_value(builder, expr)?),
                    };
                    bin = builder.ins().bs_push(spec, bin, value, size, span);
                    let next = *seg.next;
                    segment = next;
                }
                KExpr::BinaryEnd(_) => break,
                other => panic!("unexpected binary constructor segment value: {:#?}", &other),
            }
        }
        let bin = builder.ins().bs_finish(bin, span);
        builder.define_var(ret, bin);
        Ok(())
    }

    fn lower_literal<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        literal: Literal,
    ) -> anyhow::Result<Value> {
        let span = literal.span();
        match literal.value {
            Lit::Atom(value) => Ok(builder.ins().atom(value, span)),
            Lit::Integer(Int::Small(value)) => Ok(builder.ins().int(value, span)),
            Lit::Integer(Int::Big(value)) => Ok(builder.ins().bigint(value, span)),
            Lit::Float(value) => Ok(builder.ins().float(value.inner(), span)),
            Lit::Nil => Ok(builder.ins().nil(span)),
            Lit::Cons(box head, box tail) => {
                let tail = self.lower_literal(builder, tail)?;
                let head = self.lower_literal(builder, head)?;
                Ok(builder.ins().cons(head, tail, span))
            }
            Lit::Tuple(mut elements) => {
                let tup = builder.ins().tuple_imm(elements.len(), span);
                for (i, element) in elements.drain(..).enumerate() {
                    let span = element.span();
                    let value = self.lower_literal(builder, element)?;
                    builder.ins().set_element_mut(tup, i, value, span);
                }
                Ok(tup)
            }
            Lit::Map(mut lmap) => {
                let map = builder.ins().map(lmap.len(), span);
                while let Some((k, v)) = lmap.pop_first() {
                    let k = self.lower_literal(builder, k)?;
                    let v = self.lower_literal(builder, v)?;
                    builder.ins().map_put_mut(map, k, v, span);
                }
                Ok(map)
            }
            Lit::Binary(value) => Ok(builder.ins().bitstring(value, span)),
        }
    }

    fn ssa_values<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut exprs: Vec<KExpr>,
    ) -> anyhow::Result<Vec<Value>> {
        let mut values = Vec::with_capacity(exprs.len());
        for expr in exprs.drain(..) {
            values.push(self.ssa_value(builder, expr)?);
        }
        Ok(values)
    }

    fn ssa_value<'a>(&mut self, builder: &'a mut IrBuilder, expr: KExpr) -> anyhow::Result<Value> {
        match expr {
            KExpr::Var(v) => match builder.var(v.name()) {
                Some(value) => Ok(value),
                None => {
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("use of undefined variable")
                        .with_primary_label(
                            v.span(),
                            "this variable has not been defined in this scope yet",
                        )
                        .emit();
                    Err(anyhow!("invalid expression"))
                }
            },
            KExpr::Literal(lit) => self.lower_literal(builder, lit),
            expr => panic!("unexpected value expression: {:#?}", &expr),
        }
    }

    fn fail_context(&self) -> FailContext {
        use cranelift_entity::packed_option::ReservedValue;

        if !self.fail.is_reserved_value() {
            return FailContext::Guard(self.fail);
        }
        match self.landing_pads.last().copied() {
            None => FailContext::Uncaught(self.fail),
            Some(catch) => FailContext::Catch(catch),
        }
    }
}

// Select
impl<'m> LowerFunctionToSsa<'m> {
    fn select_binary<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        value: k::ValueClause,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        let src = match builder.var(var.name()) {
            Some(v) => v,
            None => panic!(
                "reference to variable `{}` that has not been defined yet",
                var.name()
            ),
        };
        let ctx_var = value
            .value
            .as_binary()
            .and_then(|b| b.segment.as_var().map(|v| v.name()))
            .unwrap();

        let (is_err, bin) = builder.ins().bs_start_match(src, span);
        builder.ins().br_if(is_err, type_fail, &[], span);
        let bin = builder.ins().cast(bin, Type::MatchContext, span);
        builder.define_var(ctx_var, bin);

        self.lower_match(builder, value_fail, *value.body)
    }

    fn select_binary_segments<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        mut values: Vec<k::ValueClause>,
        type_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();
        let mut blocks = values
            .iter()
            .skip(1)
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();
        blocks.push(type_fail);
        for (value, fail) in values.drain(..).zip(blocks.drain(..)) {
            self.select_binary_segment(builder, span, src, value, fail)?;
            builder.switch_to_block(fail);
        }
        Ok(())
    }

    fn select_binary_segment<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        src: Value,
        clause: k::ValueClause,
        fail: Block,
    ) -> anyhow::Result<()> {
        match *clause.value {
            KExpr::BinarySegment(k::BinarySegment {
                next,
                value,
                size,
                spec,
                ..
            }) => {
                let next = next.as_var().map(|v| v.name()).unwrap();
                let extracted = value.as_var().map(|v| v.name()).unwrap();
                let (extracted_value, next_value) =
                    self.select_extract_bin(builder, span, src, spec, size, fail)?;
                builder.define_var(next, next_value);
                builder.define_var(extracted, extracted_value);
                self.lower_match(builder, fail, *clause.body)
            }
            KExpr::BinaryInt(k::BinarySegment {
                next,
                value:
                    box KExpr::Literal(Literal {
                        value: Lit::Integer(Int::Small(value)),
                        ..
                    }),
                size,
                spec,
                ..
            }) => {
                let next = next.as_var().map(|v| v.name()).unwrap();
                let next_value =
                    self.select_extract_int(builder, span, src, spec, size, value, fail)?;
                builder.define_var(next, next_value);
                self.lower_match(builder, fail, *clause.body)
            }
            _ => unreachable!(),
        }
    }

    fn select_extract_bin<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        src: Value,
        spec: BinaryEntrySpecifier,
        size: Option<Box<KExpr>>,
        fail: Block,
    ) -> anyhow::Result<(Value, Value)> {
        let size = match size {
            None => None,
            Some(box sz) => Some(self.ssa_value(builder, sz)?),
        };
        let (is_err, extracted, next) = builder.ins().bs_match(spec, src, size, span);
        builder.ins().br_if(is_err, fail, &[], span);
        Ok((extracted, next))
    }

    fn select_extract_int<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        src: Value,
        spec: BinaryEntrySpecifier,
        size: Option<Box<KExpr>>,
        value: i64,
        fail: Block,
    ) -> anyhow::Result<Value> {
        let Some(size) = size.map(|box sz| self.ssa_value(builder, sz).unwrap()) else { panic!("expected size"); };
        let (is_err, next) =
            builder
                .ins()
                .bs_match_skip(spec, src, size, Immediate::I64(value), span);
        builder.ins().br_if(is_err, fail, &[], span);
        Ok(next)
    }

    fn select_binary_end<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        value: k::ValueClause,
        type_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();
        let is_err = builder.ins().bs_test_tail_imm(src, 0, span);
        builder.ins().br_if(is_err, type_fail, &[], span);
        self.lower_match(builder, type_fail, *value.body)
    }

    fn select_map<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        mut values: Vec<k::ValueClause>,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();
        let is_map = builder.ins().is_type(Type::Term(TermType::Map), src, span);
        builder.ins().br_unless(is_map, type_fail, &[], span);

        let mut blocks = values
            .iter()
            .skip(1)
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();
        blocks.push(value_fail);
        for (value, fail) in values.drain(..).zip(blocks.drain(..)) {
            let map = value.value.into_map().unwrap();
            self.select_map_value(builder, span, src, map.pairs, *value.body, fail)?;
            builder.switch_to_block(fail);
        }
        Ok(())
    }

    fn select_map_value<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        map: Value,
        mut pairs: Vec<k::MapPair>,
        body: KExpr,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        for pair in pairs.drain(..) {
            let key = self.ssa_value(builder, *pair.key)?;
            let value_var = pair.value.as_var().map(|v| v.name()).unwrap();
            let (is_err, result) = builder.ins().map_try_get(map, key, span);
            builder.ins().br_if(is_err, value_fail, &[], span);
            builder.define_var(value_var, result);
        }

        self.lower_match(builder, value_fail, body)
    }

    fn select_cons<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        value: k::ValueClause,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();
        let is_nonempty_list = builder.ins().is_type(Type::Term(TermType::Cons), src, span);
        builder
            .ins()
            .br_unless(is_nonempty_list, type_fail, &[], span);

        let cons = value.value.into_cons().unwrap();
        let list = builder.ins().cast(src, Type::Term(TermType::Cons), span);
        let (hd, tl) = builder.ins().split(list, span);
        builder.define_var(cons.head.as_var().map(|v| v.name()).unwrap(), hd);
        builder.define_var(cons.tail.as_var().map(|v| v.name()).unwrap(), tl);

        self.lower_match(builder, value_fail, *value.body)
    }

    fn select_nil<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        value: k::ValueClause,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();
        let is_nil = builder.ins().is_type(Type::Term(TermType::Nil), src, span);
        builder.ins().br_unless(is_nil, type_fail, &[], span);
        self.lower_match(builder, value_fail, *value.body)
    }

    fn select_literal<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        var: &Var,
        mut values: Vec<k::ValueClause>,
        type_fail: Block,
        value_fail: Block,
    ) -> anyhow::Result<()> {
        let src = builder.var(var.name()).unwrap();

        let mut blocks = values
            .iter()
            .skip(1)
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();
        blocks.push(type_fail);
        for (value, fail) in values.drain(..).zip(blocks.drain(..)) {
            match *value.value {
                KExpr::Literal(Literal {
                    value: Lit::Nil, ..
                }) => {
                    let is_nil = builder.ins().is_type(Type::Term(TermType::Nil), src, span);
                    builder.ins().br_unless(is_nil, fail, &[], span);
                }
                KExpr::Literal(lit) => {
                    let val = self.ssa_value(builder, KExpr::Literal(lit.clone()))?;
                    let is_eq = builder.ins().eq_exact(src, val, span);
                    builder.ins().br_unless(is_eq, fail, &[], span);
                }
                KExpr::Tuple(tuple) => {
                    let tuple_type = Type::tuple(tuple.elements.len());
                    let is_tuple = builder.ins().is_type(tuple_type.clone(), src, span);
                    builder.ins().br_unless(is_tuple, fail, &[], span);
                    let t = builder.ins().cast(src, tuple_type, span);
                    self.select_tuple_elements(builder, span, t, tuple.elements);
                }
                other => panic!("expected tuple or literal, got {:#?}", &other),
            };
            self.lower_match(builder, value_fail, *value.body)?;
            builder.switch_to_block(fail);
        }
        Ok(())
    }

    fn select_tuple_elements<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        span: SourceSpan,
        src: Value,
        elements: Vec<KExpr>,
    ) {
        for (i, element) in elements.iter().enumerate() {
            if element.annotations().contains(symbols::Unused) {
                continue;
            }
            let var = element.as_var().map(|v| v.name()).unwrap();
            let elem = builder.ins().get_element_imm(src, i, span);
            builder.define_var(var, elem);
        }
    }
}
