use anyhow::bail;
use either::Either;
use liblumen_intern::{symbols, Ident};
use liblumen_syntax_core::*;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_static_apply<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        callee: FunctionName,
        args: Vec<Value>,
        callee_span: SourceSpan,
        span: SourceSpan,
    ) -> anyhow::Result<Value> {
        let arity: u8 = args.len().try_into().unwrap();
        let callee_arity = callee.arity;
        if callee_arity != arity {
            let expected_message = format!("this function has an arity of {}", callee_arity);
            let given_message = format!("but was called with {}", arity);
            self.show_error(
                "incorrect number of arguments for callee",
                &[
                    (callee_span, expected_message.as_str()),
                    (span, given_message.as_str()),
                ],
            );
        }
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

    pub(super) fn lower_dynamic_apply<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        module: Option<Value>,
        function: Value,
        mut args: Vec<Value>,
        callee_span: SourceSpan,
        span: SourceSpan,
    ) -> anyhow::Result<Value> {
        // If the module is given, we lower to apply/3
        if let Some(module) = module {
            let apply3 = FunctionName::new(symbols::Erlang, symbols::Apply, 3);
            let nil = builder.ins().nil(span);
            let arglist = args
                .drain(0..)
                .rfold(nil, |tail, head| builder.ins().cons(head, tail, span));
            return self.lower_static_apply(
                builder,
                apply3,
                vec![module, function, arglist],
                callee_span,
                span,
            );
        }

        // Otherwise, we lower to apply/2
        let apply2 = FunctionName::new(symbols::Erlang, symbols::Apply, 2);
        let nil = builder.ins().nil(span);
        let arglist = args
            .drain(0..)
            .rfold(nil, |tail, head| builder.ins().cons(head, tail, span));
        self.lower_static_apply(builder, apply2, vec![function, arglist], callee_span, span)
    }

    pub(super) fn resolve_name<'a>(
        &mut self,
        builder: &'a IrBuilder,
        name: ast::Name,
    ) -> anyhow::Result<Either<Ident, Value>> {
        match name {
            ast::Name::Atom(n) => Ok(Either::Left(n)),
            ast::Name::Var(v) => {
                if let Some(value) = builder.get_var(v.name) {
                    return Ok(Either::Right(value));
                }
                let message = format!("variable '{}' is unbound", v);
                self.show_error(message.as_str(), &[(v.span, "used here")]);
                bail!("unable to resolve name '{}'", v)
            }
        }
    }

    pub(super) fn resolve_arity<'a>(
        &mut self,
        builder: &'a IrBuilder,
        arity: ast::Arity,
    ) -> anyhow::Result<Either<u8, Value>> {
        match arity {
            ast::Arity::Int(a) => Ok(Either::Left(a)),
            ast::Arity::Var(v) => {
                if let Some(value) = builder.get_var(v.name) {
                    return Ok(Either::Right(value));
                }
                let message = format!("variable '{}' is unbound", v);
                self.show_error(message.as_str(), &[(v.span, "used here")]);
                bail!("unable to resolve arity '{}'", v)
            }
        }
    }

    pub(super) fn resolve_local(
        &mut self,
        fun: FunctionName,
        span: SourceSpan,
    ) -> anyhow::Result<FunctionName> {
        if self.module.is_local(&fun) {
            Ok(fun.resolve(self.module.name()))
        } else if let Some(resolved) = self.module.get_import(&fun) {
            Ok(resolved)
        } else {
            let message = format!("function {} is undefined", &fun);
            self.show_error_annotated(
                message.as_str(),
                &[(span, "found here")],
                &["verify it is defined locally, or consider importing this function"],
            );
            bail!("invalid callee, no such function defined in scope")
        }
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
}
