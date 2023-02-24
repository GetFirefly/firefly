use core::ops::ControlFlow;

use paste::paste;

use crate::ast::*;

macro_rules! visit_mut_trait_method {
    ($name:ident, $ty:ty) => {
        paste! {
            fn [<visit_mut_ $name>](&mut self, [<_ $name>]: &mut $ty) -> ControlFlow<T> {
                [<visit_mut_ $name>](self, [<_ $name>])
            }
        }
    };
}

macro_rules! visit_mut_trait_impl_method {
    ($name:ident, $ty:ty) => {
        paste! {
            fn [<visit_mut_ $name>](&mut self, [<_ $name>]: &mut $ty) -> ControlFlow<T> {
                (**self).[<visit_mut_ $name>]([<_ $name>])
            }
        }
    };
}

macro_rules! visit_mut_impl_empty {
    ($name:ident, $ty:ty) => {
        paste! {
            pub fn [<visit_mut_ $name>]<V, T>(_visitor: &mut V, [<_ $name>]: &mut $ty) -> ControlFlow<T>
            where
                V: ?Sized + VisitMut<T>,
            {
                ControlFlow::Continue(())
            }
        }
    };
}

macro_rules! visitor {
    ($($name:ident => $ty:ty)*) => {
        pub trait VisitMut<T> {
            $(
                visit_mut_trait_method!($name, $ty);
            )*
        }

        impl<'a, V, T> VisitMut<T> for &'a mut V
        where
            V: ?Sized + VisitMut<T>,
        {
            $(

                visit_mut_trait_impl_method!($name, $ty);
            )*
        }
    }
}

visitor! {
    module => Module
    attribute => Literal
    record_definition => Record
    function => Function
    expr => Expr
    pattern => Expr
    fun => Fun
    anonymous_fun => AnonymousFun
    recursive_fun => RecursiveFun
    try => Try
    catch => Catch
    receive => Receive
    after => After
    case => Case
    clause => Clause
    guard => Guard
    if => If
    match => Match
    unary_expr => UnaryExpr
    binary_expr => BinaryExpr
    apply => Apply
    remote => Remote
    begin => Begin
    generator => Generator
    binary_comprehension => BinaryComprehension
    list_comprehension => ListComprehension
    record => Record
    record_access => RecordAccess
    record_index => RecordIndex
    record_update => RecordUpdate
    record_field => RecordField
    binary => Binary
    binary_pattern => Binary
    binary_element => BinaryElement
    binary_element_pattern => BinaryElement
    map => Map
    map_pattern => Map
    map_update => MapUpdate
    map_field => MapField
    tuple => Tuple
    cons => Cons
    literal => Literal
    protect => Protect
    var => Var
    function_var => FunctionVar
}

// Traversals

pub fn visit_mut_module<V, T>(visitor: &mut V, module: &mut Module) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for attr in module.attributes.values_mut() {
        visitor.visit_mut_attribute(attr)?;
    }

    for r in module.records.values_mut() {
        visitor.visit_mut_record_definition(r)?;
    }

    for f in module.functions.values_mut() {
        visitor.visit_mut_function(f)?;
    }

    ControlFlow::Continue(())
}

pub fn visit_mut_function<V, T>(visitor: &mut V, fun: &mut Function) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for (_, ref mut clause) in fun.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }

    ControlFlow::Continue(())
}

pub fn visit_mut_fun<V, T>(visitor: &mut V, fun: &mut Fun) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    match fun {
        Fun::Anonymous(ref mut f) => visitor.visit_mut_anonymous_fun(f),
        Fun::Recursive(ref mut f) => visitor.visit_mut_recursive_fun(f),
    }
}

pub fn visit_mut_recursive_fun<V, T>(visitor: &mut V, fun: &mut RecursiveFun) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for (_, ref mut clause) in fun.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_anonymous_fun<V, T>(visitor: &mut V, fun: &mut AnonymousFun) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for clause in fun.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_guard<V, T>(visitor: &mut V, guard: &mut Guard) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for expr in guard.conditions.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_expr<V, T>(visitor: &mut V, expr: &mut Expr) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    match expr {
        Expr::Var(ref mut var) => visitor.visit_mut_var(var),
        Expr::Literal(ref mut lit) => visitor.visit_mut_literal(lit),
        Expr::FunctionVar(ref mut var) => visitor.visit_mut_function_var(var),
        Expr::DelayedSubstitution(_) => ControlFlow::Continue(()),
        Expr::Cons(ref mut cons) => visitor.visit_mut_cons(cons),
        Expr::Tuple(ref mut tuple) => visitor.visit_mut_tuple(tuple),
        Expr::Map(ref mut map) => visitor.visit_mut_map(map),
        Expr::MapUpdate(ref mut up) => visitor.visit_mut_map_update(up),
        Expr::Binary(ref mut binary) => visitor.visit_mut_binary(binary),
        Expr::Record(ref mut record) => visitor.visit_mut_record(record),
        Expr::RecordAccess(ref mut access) => visitor.visit_mut_record_access(access),
        Expr::RecordIndex(ref mut index) => visitor.visit_mut_record_index(index),
        Expr::RecordUpdate(ref mut up) => visitor.visit_mut_record_update(up),
        Expr::ListComprehension(ref mut comp) => visitor.visit_mut_list_comprehension(comp),
        Expr::BinaryComprehension(ref mut comp) => visitor.visit_mut_binary_comprehension(comp),
        Expr::Generator(ref mut gen) => visitor.visit_mut_generator(gen),
        Expr::Begin(ref mut begin) => visitor.visit_mut_begin(begin),
        Expr::Apply(ref mut apply) => visitor.visit_mut_apply(apply),
        Expr::Remote(ref mut remote) => visitor.visit_mut_remote(remote),
        Expr::BinaryExpr(ref mut expr) => visitor.visit_mut_binary_expr(expr),
        Expr::UnaryExpr(ref mut expr) => visitor.visit_mut_unary_expr(expr),
        Expr::Match(ref mut expr) => visitor.visit_mut_match(expr),
        Expr::If(ref mut expr) => visitor.visit_mut_if(expr),
        Expr::Catch(ref mut expr) => visitor.visit_mut_catch(expr),
        Expr::Case(ref mut case) => visitor.visit_mut_case(case),
        Expr::Receive(ref mut receive) => visitor.visit_mut_receive(receive),
        Expr::Try(ref mut expr) => visitor.visit_mut_try(expr),
        Expr::Fun(ref mut fun) => visitor.visit_mut_fun(fun),
        Expr::Protect(ref mut protect) => visitor.visit_mut_protect(protect),
    }
}

pub fn visit_mut_pattern<V, T>(visitor: &mut V, expr: &mut Expr) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    match expr {
        Expr::Var(ref mut var) => visitor.visit_mut_var(var),
        Expr::Literal(ref mut lit) => visitor.visit_mut_literal(lit),
        Expr::DelayedSubstitution(_) => ControlFlow::Continue(()),
        Expr::Cons(ref mut cons) => visitor.visit_mut_cons(cons),
        Expr::Tuple(ref mut tuple) => visitor.visit_mut_tuple(tuple),
        Expr::Map(ref mut map) => visitor.visit_mut_map_pattern(map),
        Expr::Binary(ref mut binary) => visitor.visit_mut_binary_pattern(binary),
        Expr::Record(ref mut record) => visitor.visit_mut_record(record),
        Expr::RecordUpdate(ref mut up) => visitor.visit_mut_record_update(up),
        Expr::BinaryExpr(ref mut expr) => visitor.visit_mut_binary_expr(expr),
        Expr::UnaryExpr(ref mut expr) => visitor.visit_mut_unary_expr(expr),
        Expr::Match(ref mut expr) => visitor.visit_mut_match(expr),
        invalid => panic!("invalid pattern expression: {:?}", &invalid),
    }
}

pub fn visit_mut_cons<V, T>(visitor: &mut V, cons: &mut Cons) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(cons.head.as_mut())?;
    visitor.visit_mut_expr(cons.tail.as_mut())
}

pub fn visit_mut_tuple<V, T>(visitor: &mut V, tuple: &mut Tuple) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for element in tuple.elements.iter_mut() {
        visitor.visit_mut_expr(element)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_map<V, T>(visitor: &mut V, map: &mut Map) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for field in map.fields.iter_mut() {
        visitor.visit_mut_map_field(field)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_map_pattern<V, T>(visitor: &mut V, map: &mut Map) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for field in map.fields.iter_mut() {
        match field {
            MapField::Exact {
                ref mut key,
                ref mut value,
                ..
            } => {
                visitor.visit_mut_pattern(key)?;
                visitor.visit_mut_pattern(value)?;
            }
            _ => unreachable!(),
        }
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_map_update<V, T>(visitor: &mut V, expr: &mut MapUpdate) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.map.as_mut())?;
    for field in expr.updates.iter_mut() {
        visitor.visit_mut_map_field(field)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_map_field<V, T>(visitor: &mut V, field: &mut MapField) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    match field {
        MapField::Assoc {
            ref mut key,
            ref mut value,
            ..
        } => {
            visitor.visit_mut_expr(key)?;
            visitor.visit_mut_expr(value)
        }
        MapField::Exact {
            ref mut key,
            ref mut value,
            ..
        } => {
            visitor.visit_mut_expr(key)?;
            visitor.visit_mut_expr(value)
        }
    }
}

pub fn visit_mut_binary<V, T>(visitor: &mut V, binary: &mut Binary) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for element in binary.elements.iter_mut() {
        visitor.visit_mut_binary_element(element)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_binary_pattern<V, T>(visitor: &mut V, binary: &mut Binary) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for element in binary.elements.iter_mut() {
        visitor.visit_mut_binary_element_pattern(element)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_binary_element<V, T>(
    visitor: &mut V,
    element: &mut BinaryElement,
) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(&mut element.bit_expr)?;
    if let Some(ref mut expr) = element.bit_size {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_binary_element_pattern<V, T>(
    visitor: &mut V,
    element: &mut BinaryElement,
) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_pattern(&mut element.bit_expr)?;
    if let Some(ref mut expr) = element.bit_size {
        visitor.visit_mut_pattern(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_record<V, T>(visitor: &mut V, record: &mut Record) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for field in record.fields.iter_mut() {
        visitor.visit_mut_record_field(field)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_record_access<V, T>(visitor: &mut V, expr: &mut RecordAccess) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.record.as_mut())
}

pub fn visit_mut_record_update<V, T>(visitor: &mut V, expr: &mut RecordUpdate) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.record.as_mut())?;
    for field in expr.updates.iter_mut() {
        visitor.visit_mut_record_field(field)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_record_field<V, T>(visitor: &mut V, field: &mut RecordField) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    if let Some(ref mut expr) = field.value {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_list_comprehension<V, T>(
    visitor: &mut V,
    comp: &mut ListComprehension,
) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(comp.body.as_mut())?;
    for expr in comp.qualifiers.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_binary_comprehension<V, T>(
    visitor: &mut V,
    comp: &mut BinaryComprehension,
) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(comp.body.as_mut())?;
    for expr in comp.qualifiers.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_generator<V, T>(visitor: &mut V, gen: &mut Generator) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(gen.pattern.as_mut())?;
    visitor.visit_mut_expr(gen.expr.as_mut())
}

pub fn visit_mut_begin<V, T>(visitor: &mut V, block: &mut Begin) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for expr in block.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_apply<V, T>(visitor: &mut V, apply: &mut Apply) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(apply.callee.as_mut())?;
    for expr in apply.args.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_remote<V, T>(visitor: &mut V, remote: &mut Remote) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(remote.module.as_mut())?;
    visitor.visit_mut_expr(remote.function.as_mut())
}

pub fn visit_mut_binary_expr<V, T>(visitor: &mut V, expr: &mut BinaryExpr) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.lhs.as_mut())?;
    visitor.visit_mut_expr(expr.rhs.as_mut())
}

pub fn visit_mut_unary_expr<V, T>(visitor: &mut V, expr: &mut UnaryExpr) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.operand.as_mut())
}

pub fn visit_mut_match<V, T>(visitor: &mut V, expr: &mut Match) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_pattern(expr.pattern.as_mut())?;
    visitor.visit_mut_expr(expr.expr.as_mut())
}

pub fn visit_mut_if<V, T>(visitor: &mut V, expr: &mut If) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_catch<V, T>(visitor: &mut V, expr: &mut Catch) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(expr.expr.as_mut())
}

pub fn visit_mut_case<V, T>(visitor: &mut V, case: &mut Case) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(case.expr.as_mut())?;
    for clause in case.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_receive<V, T>(visitor: &mut V, receive: &mut Receive) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    if let Some(clauses) = receive.clauses.as_mut() {
        for clause in clauses.iter_mut() {
            visitor.visit_mut_clause(clause)?;
        }
    }
    if let Some(after) = receive.after.as_mut() {
        visitor.visit_mut_after(after)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_try<V, T>(visitor: &mut V, try_expr: &mut Try) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for expr in try_expr.exprs.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    if let Some(clauses) = try_expr.clauses.as_mut() {
        for clause in clauses.iter_mut() {
            visitor.visit_mut_clause(clause)?;
        }
    }
    if let Some(clauses) = try_expr.catch_clauses.as_mut() {
        for clause in clauses.iter_mut() {
            visitor.visit_mut_clause(clause)?;
        }
    }
    if let Some(exprs) = try_expr.after.as_mut() {
        for expr in exprs.iter_mut() {
            visitor.visit_mut_expr(expr)?;
        }
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_after<V, T>(visitor: &mut V, after: &mut After) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(after.timeout.as_mut())?;
    for expr in after.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_clause<V, T>(visitor: &mut V, clause: &mut Clause) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    for pat in clause.patterns.iter_mut() {
        visitor.visit_mut_pattern(pat)?;
    }
    for guard in clause.guards.iter_mut() {
        visitor.visit_mut_guard(guard)?;
    }
    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    ControlFlow::Continue(())
}

pub fn visit_mut_protect<V, T>(visitor: &mut V, protect: &mut Protect) -> ControlFlow<T>
where
    V: ?Sized + VisitMut<T>,
{
    visitor.visit_mut_expr(protect.body.as_mut())
}

visit_mut_impl_empty!(attribute, Literal);
visit_mut_impl_empty!(record_definition, Record);
visit_mut_impl_empty!(var, Var);
visit_mut_impl_empty!(literal, Literal);
visit_mut_impl_empty!(function_var, FunctionVar);
visit_mut_impl_empty!(record_index, RecordIndex);
