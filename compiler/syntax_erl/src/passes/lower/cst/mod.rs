use anyhow::bail;
use liblumen_diagnostics::*;
use liblumen_intern::{symbols, Ident};
use liblumen_number::Integer;
use liblumen_pass::Pass;
use liblumen_syntax_core::*;
use log::debug;

use crate::cst;
use crate::cst::Annotated;

//mod block;
mod builder;
//mod exprs;
//mod funs;
//mod helpers;
//mod patterns;
//mod receive;
//mod try_catch;

use self::builder::IrBuilder;

/// This pass is responsible for transforming the processed AST to Core IR
pub struct CstToCore {
    reporter: Reporter,
}
impl CstToCore {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for CstToCore {
    type Input<'a> = cst::Module;
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
            let name = Span::new(fun.span, *name);
            let base_visibility = if module.exports.contains(&name) {
                Visibility::PUBLIC
            } else {
                Visibility::DEFAULT
            };
            let visibility = if fun.annotations().contains(symbols::Nif) {
                base_visibility | Visibility::NIF
            } else {
                base_visibility
            };
            let mut params = vec![];
            params.resize(name.arity as usize, Type::Term(TermType::Any));
            let signature = Signature {
                visibility,
                cc: CallConv::Erlang,
                module: module.name.name,
                name: fun.name,
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
                landing_pads: vec![],
            };
            let ir_function = pass.run(fun)?;
            ir_module.define_function(ir_function);
        }

        debug!("successfully lowered ast module to core ir module");
        Ok(ir_module)
    }
}

struct LowerFunctionToCore<'m> {
    reporter: &'m mut Reporter,
    module: &'m mut Module,
    id: FuncRef,
    signature: Signature,
    landing_pads: Vec<Block>,
}
impl<'m> Pass for LowerFunctionToCore<'m> {
    type Input<'a> = cst::Fun;
    type Output<'a> = Function;

    fn run<'a>(&mut self, mut fun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
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
            fun.span,
            self.signature.clone(),
            self.module.annotations.clone(),
            self.module.signatures.clone(),
            self.module.callees.clone(),
            self.module.constants.clone(),
        );
        let mut builder = IrBuilder::new(&mut function);

        // Define the function parameters in the entry block and associate the CST vars
        // with those IR values
        let entry = builder.current_block();
        for (var, ty) in fun
            .vars
            .iter()
            .zip(builder.func.signature.params().iter().cloned())
        {
            let value = builder.func.dfg.append_block_param(entry, ty, var.span());
            builder.func.dfg.define_var(entry, var.name(), value);
        }

        self.lower_expr(&mut builder, *fun.body)?;

        debug!("LowerFunctionToCore pass completed successfully");
        Ok(function)
    }
}
impl<'m> LowerFunctionToCore<'m> {
    fn lower_expr<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: cst::Expr,
    ) -> anyhow::Result<Vec<Value>> {
        let value = match expr {
            cst::Expr::Apply(apply) => self.lower_apply(builder, apply),
            cst::Expr::Binary(bin) => self.lower_binary(builder, bin),
            cst::Expr::Call(call) => self.lower_call(builder, call),
            cst::Expr::Case(expr) => self.lower_case(builder, expr),
            cst::Expr::Catch(expr) => self.lower_catch(builder, expr),
            cst::Expr::Cons(cons) => self.lower_cons(builder, cons),
            cst::Expr::Fun(_) => todo!(),
            cst::Expr::If(expr) => self.lower_if(builder, expr),
            cst::Expr::Let(expr) => return self.lower_let(builder, expr),
            cst::Expr::LetRec(expr) => self.lower_letrec(builder, expr),
            cst::Expr::Literal(lit) => self.lower_literal(builder, lit),
            cst::Expr::Map(map) => self.lower_map(builder, map),
            cst::Expr::PrimOp(op) => self.lower_primop(builder, op),
            cst::Expr::Seq(expr) => return self.lower_seq(builder, expr),
            cst::Expr::Try(expr) => self.lower_try(builder, expr),
            cst::Expr::Tuple(tuple) => self.lower_tuple(builder, tuple),
            cst::Expr::Values(cst::Values { mut values, .. }) => {
                let mut out = Vec::with_capacity(values.len());
                for value in values.drain(..) {
                    // A values list will never have nested values
                    let mut vs = self.lower_expr(builder, value)?;
                    assert_eq!(vs.len(), 1);
                    out.push(vs.pop().unwrap());
                }
                return Ok(out);
            }
            cst::Expr::Var(var) => match builder.get_var(var.name()) {
                Some(v) => Ok(v),
                None => {
                    self.show_error("undefined variable", &[(var.span(), "used here")]);
                    bail!("undefined variable '{}'", var.name())
                }
            },
            cst::Expr::Receive(_) => {
                panic!("expected receive expressions to have been lowered to primops")
            }
            cst::Expr::Alias(_) | cst::Expr::Internal(_) => unimplemented!(),
        }?;
        Ok(vec![value])
    }

    fn lower_apply<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut apply: cst::Apply,
    ) -> anyhow::Result<Value> {
        let span = apply.span();
        let callee = self.lower_expr(builder, *apply.callee)?;
        assert_eq!(callee.len(), 1);
        let callee = callee[0];
        let mut args = Vec::with_capacity(apply.args.len());
        for arg in apply.args.drain(..) {
            let results = self.lower_expr(builder, arg)?;
            assert_eq!(results.len(), 1);
            args.push(results[0]);
        }
        let inst = builder.ins().call_indirect(callee, args.as_slice(), span);
        let (is_err, result) = {
            let results = builder.inst_results(inst);
            (results[0], results[1])
        };
        let landing_pad = self.current_landing_pad(builder);
        builder.ins().br_if(is_err, landing_pad, &[result], span);
        Ok(result)
    }

    fn lower_call<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut call: cst::Call,
    ) -> anyhow::Result<Value> {
        let span = call.span();
        let mut args = Vec::with_capacity(call.args.len());
        for arg in call.args.drain(..) {
            let results = self.lower_expr(builder, arg)?;
            assert_eq!(results.len(), 1);
            args.push(results[0]);
        }
        if let (Some(module), Some(function)) = (call.module.as_atom(), call.function.as_atom()) {
            let callee = FunctionName::new(module, function, args.len() as u8);
            let inst = if let Some(func) = builder.get_callee(callee) {
                builder.ins().call(func, args.as_slice(), span)
            } else {
                let func = builder.get_or_register_callee(callee);
                builder.ins().call(func, args.as_slice(), span)
            };
            let (is_err, result) = {
                let results = builder.inst_results(inst);
                (results[0], results[1])
            };
            let landing_pad = self.current_landing_pad(builder);
            builder.ins().br_if(is_err, landing_pad, &[result], span);
            return Ok(result);
        }
        let module = {
            let m = self.lower_expr(builder, *call.module)?;
            assert_eq!(m.len(), 1);
            m[0]
        };
        let function = {
            let mut f = self.lower_expr(builder, *call.function)?;
            assert_eq!(f.len(), 1);
            f[0]
        };
        let callee = FunctionName::new(symbols::Erlang, symbols::Apply, 3);
        let inst = if let Some(func) = builder.get_callee(callee) {
            builder.ins().call(func, args.as_slice(), span)
        } else {
            let func = builder.get_or_register_callee(callee);
            builder.ins().call(func, args.as_slice(), span)
        };
        let (is_err, result) = {
            let results = builder.inst_results(inst);
            (results[0], results[1])
        };
        let landing_pad = self.current_landing_pad(builder);
        builder.ins().br_if(is_err, landing_pad, &[result], span);
        Ok(result)
    }

    fn lower_case<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut case: cst::Case,
    ) -> anyhow::Result<Value> {
        let span = case.span();
        let args = self.lower_expr(builder, *case.arg)?;

        let current_block = builder.current_block();
        let clause_blocks = case
            .clauses
            .iter()
            .map(|_| builder.create_block())
            .collect::<Vec<_>>();
        let output_block = builder.create_block();
        let output_result =
            builder.append_block_param(output_block, Type::Term(TermType::Any), span);
        // Ensure the variable scope for the output block is the same scope as the expression
        builder.set_scope(output_block, builder.get_scope(current_block));

        for (i, clause) in case.clauses.drain(..).enumerate() {
            let next_clause_block = clause_blocks.get(i + 1).copied();
            self.lower_clause(
                builder,
                clause,
                args.as_slice(),
                next_clause_block,
                output_block,
            )?;
        }

        builder.switch_to_block(output_block);
        Ok(output_result)
    }

    fn lower_clause<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        clause: cst::Clause,
        args: &[Value],
        next_clause: Option<Block>,
        output_block: Block,
    ) -> anyhow::Result<Value> {
        assert_eq!(args.len(), clause.patterns.len());

        todo!("lower patterns into optimal chain of checks")
    }

    fn lower_literal<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        literal: cst::Literal,
    ) -> anyhow::Result<Value> {
        let span = literal.span();
        match literal.value {
            cst::Lit::Atom(value) => Ok(builder.ins().atom(value, span)),
            cst::Lit::Integer(Integer::Small(value)) => Ok(builder.ins().int(value, span)),
            cst::Lit::Integer(Integer::Big(value)) => Ok(builder.ins().bigint(value, span)),
            cst::Lit::Float(value) => Ok(builder.ins().float(value.inner(), span)),
            cst::Lit::Nil => Ok(builder.ins().nil(span)),
            cst::Lit::Cons(box head, box tail) => {
                let head = self.lower_literal(builder, head)?;
                let tail = self.lower_literal(builder, tail)?;
                Ok(builder.ins().cons(head, tail, span))
            }
            cst::Lit::Tuple(mut elements) => {
                let tup = builder.ins().tuple_imm(elements.len(), span);
                for (i, element) in elements.drain(..).enumerate() {
                    let span = element.span();
                    let value = self.lower_literal(builder, element)?;
                    let index = builder.ins().int((i + 1).try_into().unwrap(), span);
                    builder.ins().set_element(tup, index, value, span);
                }
                Ok(tup)
            }
            cst::Lit::Map(mut lmap) => {
                let map = builder.ins().map(span);
                while let Some((k, v)) = lmap.pop_first() {
                    let k = self.lower_literal(builder, k)?;
                    let v = self.lower_literal(builder, v)?;
                    builder.ins().map_put_mut(map, k, v, span);
                }
                Ok(map)
            }
            cst::Lit::Binary(value) => Ok(builder.ins().bitstring(value, span)),
        }
    }

    fn lower_cons<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        cons: cst::Cons,
    ) -> anyhow::Result<Value> {
        // Since cons cells can be deeply recursive, we avoid potential stack overflow
        // by constructing a vector of all the items, and then once we've hit the last
        // item, folding the vector into a series of cons cells
        let span = cons.span();
        let mut items = Vec::with_capacity(2);
        items.push((span, *cons.head));
        // Unroll the full set of nested cons cells
        let mut tail = *cons.tail;
        while let cst::Expr::Cons(cell) = tail {
            items.push((cell.span(), *cell.head));
            tail = *cell.tail;
        }
        // Lower expressions from right-to-left, i.e. from the bottom of the list, up
        let tail = self.lower_expr(builder, tail)?;
        assert_eq!(tail.len(), 1);
        let result = items.drain(..).try_rfold::<_, _, anyhow::Result<Value>>(
            tail[0],
            |tl, (span, hd)| {
                let hd = self.lower_expr(builder, hd)?;
                assert_eq!(hd.len(), 1);
                Ok(builder.ins().cons(hd[0], tl, span))
            },
        )?;

        Ok(result)
    }

    pub(super) fn lower_tuple<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut tuple: cst::Tuple,
    ) -> anyhow::Result<Value> {
        let span = tuple.span();

        // Allocate a tuple of the appropriate size
        let result = builder.ins().tuple_imm(tuple.elements.len(), span);
        // For each element, write the resulting value to the tuple
        for (i, element) in tuple.elements.drain(0..).enumerate() {
            let span = element.span();
            let value = self.lower_expr(builder, element)?;
            assert_eq!(value.len(), 1);
            let index = builder.ins().int((i + 1).try_into().unwrap(), span);
            builder.ins().set_element(result, index, value[0], span);
        }

        Ok(result)
    }

    pub(super) fn lower_map<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut map: cst::Map,
    ) -> anyhow::Result<Value> {
        let span = map.span();

        // Lower the map argument
        let arg = self.lower_expr(builder, *map.arg)?;
        assert_eq!(arg.len(), 1);
        let arg = arg[0];

        // Mutate the map using the given field ops
        for pair in map.pairs.drain(..) {
            let key = self.lower_expr(builder, *pair.key)?;
            assert_eq!(key.len(), 1);
            let value = self.lower_expr(builder, *pair.value)?;
            assert_eq!(value.len(), 1);
            match pair.op {
                cst::MapOp::Assoc => {
                    builder.ins().map_put_mut(arg, key[0], value[0], span);
                }
                cst::MapOp::Exact => {
                    builder.ins().map_update_mut(arg, key[0], value[0], span);
                }
            }
        }

        Ok(arg)
    }

    pub(super) fn lower_binary<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut bin: cst::Binary,
    ) -> anyhow::Result<Value> {
        let span = bin.span();

        // Special case empty binary
        if bin.segments.is_empty() {
            // Empty string constant
            return Ok(builder
                .ins()
                .binary_from_ident(Ident::new(symbols::Empty, span)));
        }

        // If we reach here, we have more than one component that need to be combined into a binary
        // The state machine for this is essentialy the following pseudocode:
        //
        // let mut builder = primop bs_init_writable
        // for element in elements {
        //   if let Err(err) = primop bits_push, builder, element {
        //     primop raise, err
        //   }
        // }
        // match primop bs_close_writable, builder {
        //   Ok(bin) => bin,
        //   Err(err) => primop raise, err
        // }

        // First, create the builder
        let bin_builder = builder.ins().bs_init_writable(span);

        // Push all elements on the builder left to right
        for segment in bin.segments.drain(..) {
            let span = segment.span();
            let value = self.lower_expr(builder, *segment.value)?;
            assert_eq!(value.len(), 1);
            let size = if let Some(box sz) = segment.size {
                let sz = self.lower_expr(builder, sz)?;
                assert_eq!(sz.len(), 1);
                Some(sz[0])
            } else {
                None
            };
            let push = builder
                .ins()
                .bs_push(segment.spec, bin_builder, value[0], size, span);
            let (is_err, result) = {
                let ins = builder.ins();
                let results = ins.data_flow_graph().inst_results(push);
                (results[0], results[1])
            };
            let landing_pad = self.current_landing_pad(builder);
            builder.ins().br_if(is_err, landing_pad, &[result], span);
        }

        // Retreive the built binary, propagating the exception if one was raised
        let close = builder.ins().bs_close_writable(bin_builder, span);
        let (is_err, result) = {
            let ins = builder.ins();
            let results = ins.data_flow_graph().inst_results(close);
            (results[0], results[1])
        };
        let landing_pad = self.current_landing_pad(builder);
        builder.ins().br_if(is_err, landing_pad, &[result], span);
        Ok(builder
            .ins()
            .cast(result, Type::Term(TermType::Bitstring), span))
    }

    pub(super) fn lower_if<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: cst::If,
    ) -> anyhow::Result<Value> {
        let span = expr.span();
        let guard = self.lower_expr(builder, *expr.guard)?;
        assert_eq!(guard.len(), 1);
        // Create blocks for the then body and control flow merge, the else flow continues in the same block
        let then_block = builder.create_block();
        let result_block = builder.create_block();
        let result = builder.append_block_param(result_block, Type::Term(TermType::Any), span);
        builder.ins().br_if(guard[0], then_block, &[], span);
        // Lower the else branch in the current block
        let else_result = self.lower_expr(builder, *expr.else_body)?;
        assert_eq!(else_result.len(), 1);
        builder.ins().br(result_block, &[else_result[0]], span);
        // Lower the then branch in its block
        builder.switch_to_block(then_block);
        let then_result = self.lower_expr(builder, *expr.then_body)?;
        assert_eq!(then_result.len(), 1);
        // Switch to the control flow join point and return the value produced by the then/else branches
        builder.ins().br(result_block, &[then_result[0]], span);
        builder.switch_to_block(result_block);
        Ok(result)
    }

    pub(super) fn lower_let<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut expr: cst::Let,
    ) -> anyhow::Result<Vec<Value>> {
        let mut values = self.lower_expr(builder, *expr.arg)?;
        assert_eq!(values.len(), expr.vars.len());
        for (value, var) in values.drain(..).zip(expr.vars.drain(..)) {
            builder.define_var(var.name(), value);
        }
        self.lower_expr(builder, *expr.body)
    }

    pub(super) fn lower_letrec<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: cst::LetRec,
    ) -> anyhow::Result<Value> {
        todo!()
    }

    pub(super) fn lower_catch<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: cst::Catch,
    ) -> anyhow::Result<Value> {
        let span = expr.span();
        // Catch requires us to split control flow such that the exceptional and non-exceptional
        // paths join back up on the return path. Exceptions should jump to a landing pad in which
        // the exception value is converted to a term
        let current_block = builder.current_block();
        // The landing pad is where control will land when an exception occurs
        // It receives a single block argument which is the exception to handle
        let landing_pad = builder.create_block();
        let exception = builder.append_block_param(landing_pad, Type::Term(TermType::Any), span);
        // The result block is where the fork in control is rejoined, it receives a single block argument which is
        // either the normal return value, or the caught/wrapped exception value
        let result_block = builder.create_block();
        let result = builder.append_block_param(result_block, Type::Term(TermType::Any), span);
        // The exit block handles wrapping exit/error reasons in the {'EXIT', Reason} tuple
        // It receives a single block argument which corresponds to `Reason` in the previous sentence.
        let exit_block = builder.create_block();
        let exit_reason = builder.append_block_param(exit_block, Type::Term(TermType::Any), span);

        // Now, in the landing pad, check what type of exception we have and branch accordingly
        builder.switch_to_block(landing_pad);
        let exception = builder.ins().cast(exception, Type::Exception, span);
        // All three types of exception require us to have the class and reason handy
        let class = builder.ins().exception_class(exception, span);
        let reason = builder.ins().exception_reason(exception, span);
        // Throws are the most common, and require no special handling, so we jump straight to the result block for them
        let is_throw = builder
            .ins()
            .eq_exact_imm(class, Immediate::Atom(symbols::Throw), span);
        builder.ins().br_if(is_throw, result_block, &[reason], span);
        // Exits are the next simplest, as we just wrap the reason in a tuple, so we jump straight to the exit block
        let is_exit = builder
            .ins()
            .eq_exact_imm(class, Immediate::Atom(symbols::Exit), span);
        builder.ins().br_if(is_exit, exit_block, &[reason], span);
        // Errors are handled in the landing pad directly
        let trace = builder.ins().exception_trace(exception, span);
        // We have to construct a new error reason, and then jump to the exit block to wrap it in the exit tuple
        let error_reason = builder.ins().tuple_imm(2, span);
        let one = builder.ins().int(1, span);
        let error_reason = builder.ins().set_element(error_reason, one, reason, span);
        let two = builder.ins().int(2, span);
        let error_reason = builder.ins().set_element(error_reason, two, trace, span);
        builder.ins().br(exit_block, &[error_reason], span);

        // In the exit block, we need just to construct the {'EXIT', Reason} tuple, and then jump to the result block
        builder.switch_to_block(exit_block);
        let wrapped_reason = builder.ins().tuple_imm(2, span);
        let wrapped_reason =
            builder
                .ins()
                .set_element_imm(wrapped_reason, 1, Immediate::Atom(symbols::EXIT), span);
        let two = builder.ins().int(2, span);
        let wrapped_reason = builder
            .ins()
            .set_element(wrapped_reason, two, exit_reason, span);
        builder.ins().br(result_block, &[wrapped_reason], span);

        // Lower the inner expression in the starting block, with the `catch` landing pad at the top of the stack
        builder.switch_to_block(current_block);
        self.landing_pads.push(landing_pad);
        let ret = self.lower_expr(builder, *expr.body)?;
        assert_eq!(ret.len(), 1);

        // We're exiting the catch now, so pop the landing pad from the stack, and branch to the result block with
        // the normal return value
        self.landing_pads.pop();
        builder.ins().br(result_block, &[ret[0]], span);

        // We leave off in the result block, using its block argument as the return value of the catch expression
        builder.switch_to_block(result_block);

        Ok(result)
    }

    pub(super) fn lower_primop<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut expr: cst::PrimOp,
    ) -> anyhow::Result<Value> {
        let span = expr.span();
        let mut args = Vec::with_capacity(expr.args.len());
        for arg in expr.args.drain(..) {
            let mut values = self.lower_expr(builder, arg)?;
            assert_eq!(values.len(), 1, "invalid use of values as primop argument");
            args.push(values[0]);
        }

        let result = match expr.name {
            symbols::RecvStart => {
                assert_eq!(
                    args.len(),
                    1,
                    "recv_start op has arity 1, but got {}",
                    args.len()
                );
                builder.ins().recv_start(args.pop().unwrap(), span)
            }
            symbols::RecvNext => {
                assert_eq!(
                    args.len(),
                    1,
                    "recv_next op has arity 1, but got {}",
                    args.len()
                );
                builder.ins().recv_next(args.pop().unwrap(), span)
            }
            symbols::RecvPeek => {
                assert_eq!(
                    args.len(),
                    1,
                    "recv_peek op has arity 1, but got {}",
                    args.len()
                );
                builder.ins().recv_peek(args.pop().unwrap(), span)
            }
            symbols::RecvPop => {
                assert_eq!(
                    args.len(),
                    1,
                    "recv_pop op has arity 1, but got {}",
                    args.len()
                );
                builder.ins().recv_pop(args.pop().unwrap(), span);
                builder.ins().atom(symbols::Ok, span)
            }
            symbols::RecvWait => {
                assert_eq!(
                    args.len(),
                    1,
                    "recv_wait op has arity 1, but got {}",
                    args.len()
                );
                builder.ins().recv_wait(args.pop().unwrap(), span);
                builder.ins().atom(symbols::Ok, span)
            }
            symbols::BitsInitWritable => {
                assert_eq!(
                    args.len(),
                    0,
                    "bits_init_writable op has arity 0, but got {}",
                    args.len()
                );
                builder.ins().bs_init_writable(span)
            }
            symbols::BitsCloseWritable => {
                assert_eq!(
                    args.len(),
                    1,
                    "bits_close_writable op has arity 1, but got {}",
                    args.len()
                );
                let bin = args.pop().unwrap();
                builder.ins().bs_close_writable(bin, span);
                bin
            }
            symbols::Raise => {
                assert_eq!(
                    args.len(),
                    3,
                    "raise op has arity 3, but got {}",
                    args.len()
                );
                let trace = args.pop().unwrap();
                let error = args.pop().unwrap();
                let class = args.pop().unwrap();
                builder.ins().raise(class, error, trace, span);
                builder.ins().none(span)
            }
            symbols::BuildStacktrace => {
                assert_eq!(
                    args.len(),
                    0,
                    "build_stacktrace op has arity 0, but got {}",
                    args.len()
                );
                builder.ins().build_stacktrace(span)
            }
            other => bail!("unknown primop {}", other),
        };
        Ok(result)
    }

    pub(super) fn lower_seq<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: cst::Seq,
    ) -> anyhow::Result<Vec<Value>> {
        let _ = self.lower_expr(builder, *expr.arg)?;
        self.lower_expr(builder, *expr.body)
    }

    pub(super) fn lower_try<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut expr: cst::Try,
    ) -> anyhow::Result<Value> {
        let span = expr.span();

        let current_block = builder.current_block();
        // The landing pad is where control will land when an exception occurs
        // It receives a single block argument which is the exception to handle
        let landing_pad = builder.create_block();
        let exception = builder.append_block_param(landing_pad, Type::Term(TermType::Any), span);
        // The result block is where the fork in control is rejoined, it receives a single block argument which is
        // either the normal return value, or the caught/wrapped exception value. Only a single value can ever be
        // produced, as implicit exports are not permitted from try
        let result_block = builder.create_block();
        let result = builder.append_block_param(result_block, Type::Term(TermType::Any), span);

        // Lower the wrapped expression, using the newly created landing pad for any thrown exceptions
        self.landing_pads.push(landing_pad);
        let args = self.lower_expr(builder, *expr.arg)?;
        self.landing_pads.pop();

        // Lower the body, binding the vars exported from the input expressions
        // Vars are exported in the body (i.e. the non-exceptional control flow path)
        // Branch to the result block with the value produced by the body
        assert_eq!(args.len(), expr.vars.len());
        for (var, arg) in expr.vars.drain(..).zip(args.iter().copied()) {
            builder.define_var(var.name(), arg);
        }
        let body = self.lower_expr(builder, *expr.body)?;
        assert_eq!(body.len(), 1);
        builder.ins().br(result_block, &[body[0]], span);

        // Lower the exception handler in the landing pad, binding the class, reason, and trace evars
        builder.switch_to_block(landing_pad);
        let exception = builder.ins().cast(exception, Type::Exception, span);
        let class = builder.ins().exception_class(exception, span);
        let reason = builder.ins().exception_reason(exception, span);
        let trace = builder.ins().exception_trace(exception, span);
        for (evar, value) in expr.evars.drain(..).zip(&[class, reason, trace]) {
            builder.define_var(evar.name(), *value);
        }

        let handled = self.lower_expr(builder, *expr.handler)?;
        assert_eq!(handled.len(), 1);
        builder.ins().br(result_block, &[handled[0]], span);

        // Switch to the result block to continue lowering
        builder.switch_to_block(result_block);

        Ok(result)
    }

    /// If there is a landing pad available on the stack, return it
    ///
    /// Otherwise, construct a default landing pad for the current function,
    /// push it on the stack, and return it.
    ///
    /// Landing pads in general are blocks which receive a single argument, the exception
    /// reference, and act on it according to the context in which they are defined:
    ///
    /// For example, the default landing pad propagates the exception to the caller by
    /// immediately returning. A bare `catch` will transfer to a block which converts the
    /// exception into its caught form and continue on to the normal return path. Statements
    /// in a `try` will all transfer to the first catch clause entry for matching.
    pub(super) fn current_landing_pad<'a>(&mut self, builder: &'a mut IrBuilder) -> Block {
        if let Some(landing_pad) = self.landing_pads.last().copied() {
            landing_pad
        } else {
            let current_block = builder.current_block();
            let landing_pad = builder.create_block();
            let span = SourceSpan::default();
            let exception =
                builder.append_block_param(landing_pad, Type::Term(TermType::Any), span);
            builder.switch_to_block(landing_pad);
            builder.ins().ret_err(exception, span);
            builder.switch_to_block(current_block);
            self.landing_pads.push(landing_pad);
            landing_pad
        }
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
