use paste::paste;

use crate::ast::*;

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
            $(
                visit_mut_trait_method!($name, $ty);
            )*
        }

        impl<'a, V> VisitMut for &'a mut V
        where
            V: ?Sized + VisitMut,
        {
            $(

                visit_mut_trait_impl_method!($name, $ty);
            )*
        }
    }
}

visitor! {
    module => Module
    attribute => UserAttribute
    record_definition => Record
    function => Function
    function_clause => FunctionClause
    expr => Expr
    pattern => Expr
    fun => Fun
    anonymous_fun => AnonymousFun
    recursive_fun => RecursiveFun
    try => Try
    try_clause => TryClause
    catch => Catch
    receive => Receive
    after => After
    case => Case
    clause => Clause
    guard => Guard
    if => If
    if_clause => IfClause
    match => Match
    unary_expr => UnaryExpr
    binary_expr => BinaryExpr
    apply => Apply
    remote => Remote
    begin => Begin
    generator => Generator
    binary_generator => BinaryGenerator
    binary_comprehension => BinaryComprehension
    list_comprehension => ListComprehension
    record => Record
    record_access => RecordAccess
    record_index => RecordIndex
    record_update => RecordUpdate
    record_field => RecordField
    binary => Binary
    binary_element => BinaryElement
    map => Map
    map_update => MapUpdate
    map_projection => MapProjection
    map_field => MapField
    tuple => Tuple
    cons => Cons
    literal => Literal
    var => Var
    function_name => FunctionName
}

// Traversals

pub fn visit_mut_module<V: ?Sized + VisitMut>(
    visitor: &mut V,
    module: &mut Module,
) -> anyhow::Result<()> {
    for attr in module.attributes.values_mut() {
        visitor.visit_mut_attribute(attr)?;
    }

    for r in module.records.values_mut() {
        visitor.visit_mut_record_definition(r)?;
    }

    for f in module.functions.values_mut() {
        visitor.visit_mut_function(f)?;
    }

    Ok(())
}

pub fn visit_mut_function<V: ?Sized + VisitMut>(
    visitor: &mut V,
    fun: &mut Function,
) -> anyhow::Result<()> {
    for clause in fun.clauses.iter_mut() {
        visitor.visit_mut_function_clause(clause)?;
    }

    Ok(())
}

pub fn visit_mut_fun<V: ?Sized + VisitMut>(visitor: &mut V, fun: &mut Fun) -> anyhow::Result<()> {
    match fun {
        Fun::Anonymous(ref mut f) => visitor.visit_mut_anonymous_fun(f),
        Fun::Recursive(ref mut f) => visitor.visit_mut_recursive_fun(f),
    }
}

pub fn visit_mut_recursive_fun<V: ?Sized + VisitMut>(
    visitor: &mut V,
    fun: &mut RecursiveFun,
) -> anyhow::Result<()> {
    for clause in fun.clauses.iter_mut() {
        visitor.visit_mut_function_clause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_anonymous_fun<V: ?Sized + VisitMut>(
    visitor: &mut V,
    fun: &mut AnonymousFun,
) -> anyhow::Result<()> {
    for clause in fun.clauses.iter_mut() {
        visitor.visit_mut_function_clause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_function_clause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut FunctionClause,
) -> anyhow::Result<()> {
    for param in clause.params.iter_mut() {
        visitor.visit_mut_pattern(param)?;
    }

    if let Some(guards) = clause.guard.as_mut() {
        for guard in guards.iter_mut() {
            visitor.visit_mut_guard(guard)?;
        }
    }

    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }

    Ok(())
}

pub fn visit_mut_guard<V: ?Sized + VisitMut>(
    visitor: &mut V,
    guard: &mut Guard,
) -> anyhow::Result<()> {
    for expr in guard.conditions.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_expr<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Expr,
) -> anyhow::Result<()> {
    match expr {
        Expr::Var(ref mut var) => visitor.visit_mut_var(var),
        Expr::Literal(ref mut lit) => visitor.visit_mut_literal(lit),
        Expr::FunctionName(ref mut name) => visitor.visit_mut_function_name(name),
        Expr::DelayedSubstitution(_, _) => Ok(()),
        Expr::Nil(_) => Ok(()),
        Expr::Cons(ref mut cons) => visitor.visit_mut_cons(cons),
        Expr::Tuple(ref mut tuple) => visitor.visit_mut_tuple(tuple),
        Expr::Map(ref mut map) => visitor.visit_mut_map(map),
        Expr::MapUpdate(ref mut up) => visitor.visit_mut_map_update(up),
        Expr::MapProjection(ref mut proj) => visitor.visit_mut_map_projection(proj),
        Expr::Binary(ref mut binary) => visitor.visit_mut_binary(binary),
        Expr::Record(ref mut record) => visitor.visit_mut_record(record),
        Expr::RecordAccess(ref mut access) => visitor.visit_mut_record_access(access),
        Expr::RecordIndex(ref mut index) => visitor.visit_mut_record_index(index),
        Expr::RecordUpdate(ref mut up) => visitor.visit_mut_record_update(up),
        Expr::ListComprehension(ref mut comp) => visitor.visit_mut_list_comprehension(comp),
        Expr::BinaryComprehension(ref mut comp) => visitor.visit_mut_binary_comprehension(comp),
        Expr::Generator(ref mut gen) => visitor.visit_mut_generator(gen),
        Expr::BinaryGenerator(ref mut gen) => visitor.visit_mut_binary_generator(gen),
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
    }
}

pub fn visit_mut_pattern<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Expr,
) -> anyhow::Result<()> {
    match expr {
        Expr::Var(ref mut var) => visitor.visit_mut_var(var),
        Expr::Literal(ref mut lit) => visitor.visit_mut_literal(lit),
        Expr::DelayedSubstitution(_, _) => Ok(()),
        Expr::Nil(_) => Ok(()),
        Expr::Cons(ref mut cons) => visitor.visit_mut_cons(cons),
        Expr::Tuple(ref mut tuple) => visitor.visit_mut_tuple(tuple),
        Expr::Map(ref mut map) => visitor.visit_mut_map(map),
        Expr::MapProjection(ref mut proj) => visitor.visit_mut_map_projection(proj),
        Expr::Binary(ref mut binary) => visitor.visit_mut_binary(binary),
        Expr::Record(ref mut record) => visitor.visit_mut_record(record),
        Expr::RecordUpdate(ref mut up) => visitor.visit_mut_record_update(up),
        Expr::BinaryExpr(ref mut expr) => visitor.visit_mut_binary_expr(expr),
        Expr::UnaryExpr(ref mut expr) => visitor.visit_mut_unary_expr(expr),
        invalid => panic!("invalid pattern expression: {:?}", &invalid),
    }
}

pub fn visit_mut_cons<V: ?Sized + VisitMut>(
    visitor: &mut V,
    cons: &mut Cons,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(cons.head.as_mut())?;
    visitor.visit_mut_expr(cons.tail.as_mut())
}

pub fn visit_mut_tuple<V: ?Sized + VisitMut>(
    visitor: &mut V,
    tuple: &mut Tuple,
) -> anyhow::Result<()> {
    for element in tuple.elements.iter_mut() {
        visitor.visit_mut_expr(element)?;
    }
    Ok(())
}

pub fn visit_mut_map<V: ?Sized + VisitMut>(visitor: &mut V, map: &mut Map) -> anyhow::Result<()> {
    for field in map.fields.iter_mut() {
        visitor.visit_mut_map_field(field)?;
    }
    Ok(())
}

pub fn visit_mut_map_update<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut MapUpdate,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.map.as_mut())?;
    for field in expr.updates.iter_mut() {
        visitor.visit_mut_map_field(field)?;
    }
    Ok(())
}

pub fn visit_mut_map_projection<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut MapProjection,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.map.as_mut())?;
    for field in expr.fields.iter_mut() {
        visitor.visit_mut_map_field(field)?;
    }
    Ok(())
}

pub fn visit_mut_map_field<V: ?Sized + VisitMut>(
    visitor: &mut V,
    field: &mut MapField,
) -> anyhow::Result<()> {
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

pub fn visit_mut_binary<V: ?Sized + VisitMut>(
    visitor: &mut V,
    binary: &mut Binary,
) -> anyhow::Result<()> {
    for element in binary.elements.iter_mut() {
        visitor.visit_mut_binary_element(element)?;
    }
    Ok(())
}

pub fn visit_mut_binary_element<V: ?Sized + VisitMut>(
    visitor: &mut V,
    element: &mut BinaryElement,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(&mut element.bit_expr)?;
    if let Some(ref mut expr) = element.bit_size {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_record<V: ?Sized + VisitMut>(
    visitor: &mut V,
    record: &mut Record,
) -> anyhow::Result<()> {
    for field in record.fields.iter_mut() {
        visitor.visit_mut_record_field(field)?;
    }
    Ok(())
}

pub fn visit_mut_record_access<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut RecordAccess,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.record.as_mut())
}

pub fn visit_mut_record_update<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut RecordUpdate,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.record.as_mut())?;
    for field in expr.updates.iter_mut() {
        visitor.visit_mut_record_field(field)?;
    }
    Ok(())
}

pub fn visit_mut_record_field<V: ?Sized + VisitMut>(
    visitor: &mut V,
    field: &mut RecordField,
) -> anyhow::Result<()> {
    if let Some(ref mut expr) = field.value {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_list_comprehension<V: ?Sized + VisitMut>(
    visitor: &mut V,
    comp: &mut ListComprehension,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(comp.body.as_mut())?;
    for expr in comp.qualifiers.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_binary_comprehension<V: ?Sized + VisitMut>(
    visitor: &mut V,
    comp: &mut BinaryComprehension,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(comp.body.as_mut())?;
    for expr in comp.qualifiers.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_generator<V: ?Sized + VisitMut>(
    visitor: &mut V,
    gen: &mut Generator,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(gen.pattern.as_mut())?;
    visitor.visit_mut_expr(gen.expr.as_mut())
}

pub fn visit_mut_binary_generator<V: ?Sized + VisitMut>(
    visitor: &mut V,
    gen: &mut BinaryGenerator,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(gen.pattern.as_mut())?;
    visitor.visit_mut_expr(gen.expr.as_mut())
}

pub fn visit_mut_begin<V: ?Sized + VisitMut>(
    visitor: &mut V,
    block: &mut Begin,
) -> anyhow::Result<()> {
    for expr in block.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_apply<V: ?Sized + VisitMut>(
    visitor: &mut V,
    apply: &mut Apply,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(apply.callee.as_mut())?;
    for expr in apply.args.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_remote<V: ?Sized + VisitMut>(
    visitor: &mut V,
    remote: &mut Remote,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(remote.module.as_mut())?;
    visitor.visit_mut_expr(remote.function.as_mut())
}

pub fn visit_mut_binary_expr<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut BinaryExpr,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.lhs.as_mut())?;
    visitor.visit_mut_expr(expr.rhs.as_mut())
}

pub fn visit_mut_unary_expr<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut UnaryExpr,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.operand.as_mut())
}

pub fn visit_mut_match<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Match,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.pattern.as_mut())?;
    visitor.visit_mut_expr(expr.expr.as_mut())
}

pub fn visit_mut_if<V: ?Sized + VisitMut>(visitor: &mut V, expr: &mut If) -> anyhow::Result<()> {
    for clause in expr.clauses.iter_mut() {
        visitor.visit_mut_if_clause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_if_clause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut IfClause,
) -> anyhow::Result<()> {
    for guard in clause.guards.iter_mut() {
        visitor.visit_mut_guard(guard)?;
    }
    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_catch<V: ?Sized + VisitMut>(
    visitor: &mut V,
    expr: &mut Catch,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(expr.expr.as_mut())
}

pub fn visit_mut_case<V: ?Sized + VisitMut>(
    visitor: &mut V,
    case: &mut Case,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(case.expr.as_mut())?;
    for clause in case.clauses.iter_mut() {
        visitor.visit_mut_clause(clause)?;
    }
    Ok(())
}

pub fn visit_mut_receive<V: ?Sized + VisitMut>(
    visitor: &mut V,
    receive: &mut Receive,
) -> anyhow::Result<()> {
    if let Some(clauses) = receive.clauses.as_mut() {
        for clause in clauses.iter_mut() {
            visitor.visit_mut_clause(clause)?;
        }
    }
    if let Some(after) = receive.after.as_mut() {
        visitor.visit_mut_after(after)?;
    }
    Ok(())
}

pub fn visit_mut_try<V: ?Sized + VisitMut>(
    visitor: &mut V,
    try_expr: &mut Try,
) -> anyhow::Result<()> {
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
            visitor.visit_mut_try_clause(clause)?;
        }
    }
    if let Some(exprs) = try_expr.after.as_mut() {
        for expr in exprs.iter_mut() {
            visitor.visit_mut_expr(expr)?;
        }
    }
    Ok(())
}

pub fn visit_mut_try_clause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut TryClause,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(&mut clause.error)?;
    if let Some(guards) = clause.guard.as_mut() {
        for guard in guards.iter_mut() {
            visitor.visit_mut_guard(guard)?;
        }
    }
    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_after<V: ?Sized + VisitMut>(
    visitor: &mut V,
    after: &mut After,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(after.timeout.as_mut())?;
    for expr in after.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

pub fn visit_mut_clause<V: ?Sized + VisitMut>(
    visitor: &mut V,
    clause: &mut Clause,
) -> anyhow::Result<()> {
    visitor.visit_mut_expr(&mut clause.pattern)?;
    if let Some(guards) = clause.guard.as_mut() {
        for guard in guards.iter_mut() {
            visitor.visit_mut_guard(guard)?;
        }
    }
    for expr in clause.body.iter_mut() {
        visitor.visit_mut_expr(expr)?;
    }
    Ok(())
}

visit_mut_impl_empty!(attribute, UserAttribute);
visit_mut_impl_empty!(record_definition, Record);
visit_mut_impl_empty!(var, Var);
visit_mut_impl_empty!(literal, Literal);
visit_mut_impl_empty!(function_name, FunctionName);
visit_mut_impl_empty!(record_index, RecordIndex);
