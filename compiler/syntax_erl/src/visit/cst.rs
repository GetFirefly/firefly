use paste::paste;

use liblumen_intern::Ident;

use crate::cst::*;

macro_rules! visit_mut_trait_method {
    ($name:ident, $ty:ty) => {
        paste! {
            fn [<visit_mut_ $name>](&mut self, [<_ $name>]: &mut $ty) -> anyhow::Result<()> {
                [<visit_mut_ $name>](self, [<_ $name>])
            }
        }
    };
}

macro_rules! visit_mut_trait_impl_method {
    ($name:ident, $ty:ty) => {
        paste! {
            fn [<visit_mut_ $name>](&mut self, [<_ $name>]: &mut $ty) -> anyhow::Result<()> {
                (**self).[<visit_mut_ $name>]([<_ $name>])
            }
        }
    };
}

macro_rules! visit_mut_impl_empty {
    ($name:ident, $ty:ty) => {
        paste! {
            pub fn [<visit_mut_ $name>]<V: ?Sized + VisitMut>(_visitor: &mut V, [<_ $name>]: &mut $ty) -> anyhow::Result<()> {
                Ok(())
            }
        }
    };
}

macro_rules! visitor {
    ($($name:ident => $ty:ty)*) => {
        pub trait VisitMut {
            fn visit_mut_attribute(&mut self, name: Ident, value: &mut Expr) -> anyhow::Result<()> {
                visit_mut_attribute(self, name, value)
            }

            $(
                visit_mut_trait_method!($name, $ty);
            )*
        }

        impl<'a, V> VisitMut for &'a mut V
        where
            V: ?Sized + VisitMut,
        {
            fn visit_mut_attribute(&mut self, name: Ident, value: &mut Expr) -> anyhow::Result<()> {
                (**self).visit_mut_attribute(name, value)
            }

            $(

                visit_mut_trait_impl_method!($name, $ty);
            )*
        }
    }
}

visitor! {
    module => Module
    alias => Alias
    apply => Apply
    binary => Binary
    ibinary => IBinary
    bitstring => Bitstring
    ibitstring => IBitstring
    call => Call
    case => Case
    icase => ICase
    catch => Catch
    icatch => ICatch
    clause => Clause
    iclause => IClause
    cons => Cons
    expr => Expr
    iexpr => IExpr
    iexprs => IExprs
    fun => Fun
    ifun => IFun
    guard => Expr
    if => If
    iif => IIf
    let => Let
    letrec => LetRec
    iletrec => ILetRec
    def => (Var, Expr)
    literal => Literal
    map => Map
    imatch => IMatch
    pattern => Expr
    primop => PrimOp
    iprotect => IProtect
    receive => Receive
    ireceive1 => IReceive1
    ireceive2 => IReceive2
    seq => Seq
    iset => ISet
    try => Try
    itry => ITry
    tuple => Tuple
    values => Values
    var => Var
}

// Traversals

pub fn visit_mut_module<V: ?Sized + VisitMut>(
    visitor: &mut V,
    module: &mut Module,
) -> anyhow::Result<()> {
    for (id, value) in module.attributes.iter_mut() {
        visitor.visit_mut_attribute(*id, value)?;
    }

    for f in module.functions.values_mut() {
        visitor.visit_mut_fun(f)?;
    }

    Ok(())
}

pub fn visit_mut_alias<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Alias,
) -> anyhow::Result<()> {
    visitor.visit_mut_var(&mut expr.var)?;
    visitor.visit_mut_pattern(expr.pattern.as_mut())
}

pub fn visit_mut_apply<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Apply,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.callee.as_mut())?;
    for arg in expr.args.iter_mut() {
        visitor.visit_mut_expr(arg)?;
    }
    Ok(())
}

pub fn visit_mut_binary<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Binary,
) -> anyhow::Result<()> {
    for segment in expr.segments.iter_mut() {
        visitor.visit_mut_bitstring(segment)?;
    }
    Ok(())
}

pub fn visit_mut_ibinary<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IBinary,
) -> anyhow::Result<()> {
    for segment in expr.segments.iter_mut() {
        visitor.visit_mut_ibitstring(segment)?;
    }
    Ok(())
}

pub fn visit_mut_bitstring<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Bitstring,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.value.as_mut())?;
    if let Some(size) = expr.size.as_mut() {
        visitor.visit_mut_expr(size)?;
    }
    Ok(())
}

pub fn visit_mut_ibitstring<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IBitstring,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.value.as_mut())?;
    for expr in expr.size.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_call<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Call,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.module.as_mut())?;
    visitor.visit_mut_expr(expr.function.as_mut())?;
    for arg in expr.args.iter_mut() {
        visitor.visit_mut_expr(arg)?;
    }
    Ok(())
}

pub fn visit_mut_case<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Case,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.arg.as_mut())?;
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_icase<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut ICase,
) -> anyhow::Result<()> {
    for arg in expr.args.iter_mut() {
        visitor.visit_mut_expr(arg)?;
    }
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_iclause(clause)?;
    }
    visitor.visit_mut_iclause(expr.fail.as_mut())
}

pub fn visit_mut_catch<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Catch,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.body.as_mut())
}

pub fn visit_mut_icatch<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut ICatch,
) -> anyhow::Result<()> {
    for expr in expr.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_clause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut Clause,
) -> anyhow::Result<()> {
    for pat in clause.patterns.iter_mut() {
        visitor.visit_mut_pattern(pat)?;
    }
    if let Some(guard) = clause.guard.as_mut() {
        visitor.visit_mut_guard(guard)?;
    }
    visitor.visit_mut_expr(clause.body.as_mut())
}

pub fn visit_mut_iclause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut IClause,
) -> anyhow::Result<()> {
    for pat in clause.patterns.iter_mut() {
        visitor.visit_mut_pattern(pat)?;
    }
    for guard in clause.guards.iter_mut() {
        visitor.visit_mut_guard(guard)?;
    }
    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_guard<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Expr,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr)
}

pub fn visit_mut_cons<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Cons,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.head.as_mut())?;
    visitor.visit_mut_expr(expr.tail.as_mut())
}

pub fn visit_mut_iexprs<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IExprs,
) -> anyhow::Result<()> {
    for exprs in expr.bodies.iter_mut() {
        for expr in exprs.iter_mut() {
            visitor.visit_mut_expr(expr)?;
        }
    }
    Ok(())
}

pub fn visit_mut_fun<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut Fun) -> anyhow::Result<()> {
    for var in expr.vars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }

    visitor.visit_mut_expr(expr.body.as_mut())
}

pub fn visit_mut_ifun<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IFun,
) -> anyhow::Result<()> {
    for var in expr.vars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }

    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_iclause(clause)?;
    }

    visitor.visit_mut_iclause(expr.fail.as_mut())
}

pub fn visit_mut_if<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut If) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.guard.as_mut())?;
    visitor.visit_mut_expr(expr.then_body.as_mut())?;
    visitor.visit_mut_expr(expr.else_body.as_mut())
}

pub fn visit_mut_iif<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut IIf) -> anyhow::Result<()> {
    for expr in expr.guards.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    for expr in expr.then_body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    for expr in expr.else_body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_let<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut Let) -> anyhow::Result<()> {
    for var in expr.vars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }

    visitor.visit_mut_expr(expr.arg.as_mut())?;
    visitor.visit_mut_expr(expr.body.as_mut())
}

pub fn visit_mut_letrec<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut LetRec,
) -> anyhow::Result<()> {
    for def in expr.defs.iter_mut() {
        visitor.visit_mut_def(def)?;
    }

    visitor.visit_mut_expr(expr.body.as_mut())
}

pub fn visit_mut_iletrec<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut ILetRec,
) -> anyhow::Result<()> {
    for def in expr.defs.iter_mut() {
        visitor.visit_mut_def(def)?;
    }

    for expr in expr.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_def<V: ?Sized + VisitMut>(
    visitor: &mut V,
    def: &mut (Var, Expr),
) -> anyhow::Result<()> {
    visitor.visit_mut_var(&mut def.0)?;
    visitor.visit_mut_expr(&mut def.1)
}

pub fn visit_mut_map<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut Map) -> anyhow::Result<()> {
    if expr.is_pattern {
        visitor.visit_mut_pattern(expr.arg.as_mut())?;
        for pair in expr.pairs.iter_mut() {
            visitor.visit_mut_pattern(&mut pair.key)?;
            visitor.visit_mut_pattern(&mut pair.value)?;
        }
    } else {
        visitor.visit_mut_expr(expr.arg.as_mut())?;
        for pair in expr.pairs.iter_mut() {
            visitor.visit_mut_expr(&mut pair.key)?;
            visitor.visit_mut_expr(&mut pair.value)?;
        }
    }
    Ok(())
}

pub fn visit_mut_imatch<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IMatch,
) -> anyhow::Result<()> {
    visitor.visit_mut_pattern(expr.pattern.as_mut())?;
    for guard in expr.guards.iter_mut() {
        visitor.visit_mut_guard(guard)?;
    }
    visitor.visit_mut_expr(expr.arg.as_mut())?;
    visitor.visit_mut_iclause(expr.fail.as_mut())
}

pub fn visit_mut_iprotect<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IProtect,
) -> anyhow::Result<()> {
    for expr in expr.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_primop<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut PrimOp,
) -> anyhow::Result<()> {
    for arg in expr.args.iter_mut() {
        visitor.visit_mut_expr(arg)?;
    }
    Ok(())
}

pub fn visit_mut_receive<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Receive,
) -> anyhow::Result<()> {
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    visitor.visit_mut_expr(expr.timeout.as_mut())?;
    visitor.visit_mut_expr(expr.action.as_mut())
}

pub fn visit_mut_ireceive1<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IReceive1,
) -> anyhow::Result<()> {
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_iclause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_ireceive2<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IReceive2,
) -> anyhow::Result<()> {
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_iclause(clause)?;
    }
    visitor.visit_mut_expr(expr.timeout.as_mut())?;
    for expr in expr.action.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_seq<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut Seq) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.arg.as_mut())?;
    visitor.visit_mut_expr(expr.body.as_mut())
}

pub fn visit_mut_iset<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut ISet,
) -> anyhow::Result<()> {
    visitor.visit_mut_var(&mut expr.var)?;
    visitor.visit_mut_expr(expr.arg.as_mut())
}

pub fn visit_mut_try<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut Try) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.arg.as_mut())?;
    for var in expr.vars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }
    visitor.visit_mut_expr(expr.body.as_mut())?;
    for var in expr.evars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }
    visitor.visit_mut_expr(expr.handler.as_mut())
}

pub fn visit_mut_itry<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut ITry,
) -> anyhow::Result<()> {
    for arg in expr.args.iter_mut() {
        visitor.visit_mut_expr(arg)?;
    }
    for var in expr.vars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }
    for expr in expr.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    for var in expr.evars.iter_mut() {
        visitor.visit_mut_var(var)?;
    }
    visitor.visit_mut_expr(expr.handler.as_mut())
}

pub fn visit_mut_tuple<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Tuple,
) -> anyhow::Result<()> {
    for element in expr.elements.iter_mut() {
        visitor.visit_mut_expr(element)?;
    }
    Ok(())
}

pub fn visit_mut_values<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Values,
) -> anyhow::Result<()> {
    for value in expr.values.iter_mut() {
        visitor.visit_mut_expr(value)?;
    }
    Ok(())
}

pub fn visit_mut_expr<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Expr,
) -> anyhow::Result<()> {
    match expr {
        Expr::Alias(ref mut expr) => visitor.visit_mut_alias(expr),
        Expr::Apply(ref mut expr) => visitor.visit_mut_apply(expr),
        Expr::Binary(ref mut expr) => visitor.visit_mut_binary(expr),
        Expr::Call(ref mut expr) => visitor.visit_mut_call(expr),
        Expr::Case(ref mut expr) => visitor.visit_mut_case(expr),
        Expr::Catch(ref mut expr) => visitor.visit_mut_catch(expr),
        Expr::Cons(ref mut expr) => visitor.visit_mut_cons(expr),
        Expr::Fun(ref mut expr) => visitor.visit_mut_fun(expr),
        Expr::If(ref mut expr) => visitor.visit_mut_if(expr),
        Expr::Let(ref mut expr) => visitor.visit_mut_let(expr),
        Expr::LetRec(ref mut expr) => visitor.visit_mut_letrec(expr),
        Expr::Literal(ref mut expr) => visitor.visit_mut_literal(expr),
        Expr::Map(ref mut expr) => visitor.visit_mut_map(expr),
        Expr::PrimOp(ref mut expr) => visitor.visit_mut_primop(expr),
        Expr::Receive(ref mut expr) => visitor.visit_mut_receive(expr),
        Expr::Seq(ref mut expr) => visitor.visit_mut_seq(expr),
        Expr::Try(ref mut expr) => visitor.visit_mut_try(expr),
        Expr::Tuple(ref mut expr) => visitor.visit_mut_tuple(expr),
        Expr::Values(ref mut expr) => visitor.visit_mut_values(expr),
        Expr::Var(ref mut expr) => visitor.visit_mut_var(expr),
        Expr::Internal(ref mut iexpr) => visitor.visit_mut_iexpr(iexpr),
    }
}

pub fn visit_mut_iexpr<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut IExpr,
) -> anyhow::Result<()> {
    match expr {
        IExpr::Binary(ref mut expr) => visitor.visit_mut_ibinary(expr),
        IExpr::Case(ref mut expr) => visitor.visit_mut_icase(expr),
        IExpr::Catch(ref mut expr) => visitor.visit_mut_icatch(expr),
        IExpr::Exprs(ref mut expr) => visitor.visit_mut_iexprs(expr),
        IExpr::Fun(ref mut expr) => visitor.visit_mut_ifun(expr),
        IExpr::If(ref mut expr) => visitor.visit_mut_iif(expr),
        IExpr::LetRec(ref mut expr) => visitor.visit_mut_iletrec(expr),
        IExpr::Match(ref mut expr) => visitor.visit_mut_imatch(expr),
        IExpr::Protect(ref mut expr) => visitor.visit_mut_iprotect(expr),
        IExpr::Receive1(ref mut expr) => visitor.visit_mut_ireceive1(expr),
        IExpr::Receive2(ref mut expr) => visitor.visit_mut_ireceive2(expr),
        IExpr::Set(ref mut expr) => visitor.visit_mut_iset(expr),
        IExpr::Simple(ref mut expr) => visitor.visit_mut_expr(expr.as_mut()),
        IExpr::Try(ref mut expr) => visitor.visit_mut_itry(expr),
    }
}

pub fn visit_mut_pattern<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Expr,
) -> anyhow::Result<()> {
    match expr {
        Expr::Alias(ref mut expr) => visitor.visit_mut_alias(expr),
        Expr::Binary(ref mut expr) => visitor.visit_mut_binary(expr),
        Expr::Cons(ref mut expr) => visitor.visit_mut_cons(expr),
        Expr::Literal(ref mut expr) => visitor.visit_mut_literal(expr),
        Expr::Map(ref mut expr) => visitor.visit_mut_map(expr),
        Expr::Tuple(ref mut expr) => visitor.visit_mut_tuple(expr),
        Expr::Values(ref mut expr) => visitor.visit_mut_values(expr),
        Expr::Var(ref mut expr) => visitor.visit_mut_var(expr),
        Expr::Internal(ref mut iexpr) => visitor.visit_mut_iexpr(iexpr),
        invalid => panic!("invalid pattern expression: {:?}", &invalid),
    }
}

pub fn visit_mut_attribute<V: ?Sized + VisitMut>(
    _visitor: &mut V,
    _name: Ident,
    _value: &mut Expr,
) -> anyhow::Result<()> {
    Ok(())
}

visit_mut_impl_empty!(var, Var);
visit_mut_impl_empty!(literal, Literal);
