///! Immutable/mutable visitors for Erlang AST

/// Converts AST to a pretty printed source string.
pub mod pretty_print;

use std::collections::HashSet;

use super::ast::*;

pub use self::pretty_print::PrettyPrintVisitor;

macro_rules! make_visitor {
    ($visitor_trait_name:ident, $($mutability:ident)*) => {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        pub trait $visitor_trait_name<'ast> {
            // Override the following functions. The `walk` functions are the default behavior.

            /// This is the initial function used to start traversing the AST. By default, this
            /// will simply recursively walk down the AST without performing any meaningful action.
            fn visit(&mut self, module: &'ast $($mutability)* Module) {
                self.walk_module(module);
            }

            fn visit_imports(&mut self, _imports: &'ast $($mutability)* HashSet<ResolvedFunctionName>) {}
            fn visit_exports(&mut self, _exports: &'ast $($mutability)* HashSet<ResolvedFunctionName>) {}
            fn visit_exported_types(&mut self, _exported_types: &'ast $($mutability)* HashSet<ResolvedFunctionName>) {}
            fn visit_behaviours(&mut self, _behaviours: &'ast $($mutability)* HashSet<Ident>) {}

            fn visit_type_def(&mut self, type_def: &'ast $($mutability)* TypeDef) {
                self.walk_type_def(type_def);
            }

            fn visit_record(&mut self, record: &'ast $($mutability)* Record) {
                self.walk_record(record);
            }

            fn visit_record_field(&mut self, field: &'ast $($mutability)* RecordField) {
                self.walk_record_field(field);
            }

            fn visit_callback(&mut self, callback: &'ast $($mutability)* Callback) {
                self.walk_callback(callback);
            }

            fn visit_callback_signature(&mut self, sig: &'ast $($mutability)* TypeSig) {
                self.walk_sig(sig);
            }

            fn visit_spec(&mut self, spec: &'ast $($mutability)* TypeSpec) {
                self.walk_spec(spec);
            }

            fn visit_spec_signature(&mut self, sig: &'ast $($mutability)* TypeSig) {
                self.walk_sig(sig);
            }

            fn visit_attribute(&mut self, attribute: &'ast $($mutability)* UserAttribute) {
                self.walk_attribute(attribute);
            }

            fn visit_function(&mut self, function: &'ast $($mutability)* NamedFunction) {
                self.walk_function(function);
            }

            fn visit_function_clause(&mut self, clause: &'ast $($mutability)* FunctionClause) {
                self.walk_function_clause(clause);
            }

            fn visit_guard(&mut self, guard: &'ast $($mutability)* Guard) {
                self.walk_guard(guard);
            }

            fn visit_type(&mut self, spec: &'ast $($mutability)* Type) {
                self.walk_type(spec);
            }

            fn visit_type_guard(&mut self, guard: &'ast $($mutability)* TypeGuard) {
                self.walk_type_guard(guard);
            }

            fn visit_expression(&mut self, expr: &'ast $($mutability)* Expr) {
                match expr {
                    &$($mutability)* Expr::Var(ref $($mutability)* expr) => self.visit_identifier(expr),
                    &$($mutability)* Expr::Literal(ref $($mutability)* expr) => self.visit_literal(expr),
                    &$($mutability)* Expr::FunctionName(ref $($mutability)* expr) => self.visit_function_name(expr),
                    &$($mutability)* Expr::Nil(ref $($mutability)* expr) => self.visit_nil(expr),
                    &$($mutability)* Expr::Cons(ref $($mutability)* expr) => self.visit_cons(expr),
                    &$($mutability)* Expr::Tuple(ref $($mutability)* expr) => self.visit_tuple(expr),
                    &$($mutability)* Expr::Map(ref $($mutability)* expr) => self.visit_map(expr),
                    &$($mutability)* Expr::MapUpdate(ref $($mutability)* expr) => self.visit_map_update(expr),
                    &$($mutability)* Expr::MapProjection(ref $($mutability)* expr) => self.visit_map_projection(expr),
                    &$($mutability)* Expr::Binary(ref $($mutability)* expr) => self.visit_binary(expr),
                    &$($mutability)* Expr::Record(ref $($mutability)* expr) => self.visit_record(expr),
                    &$($mutability)* Expr::RecordAccess(ref $($mutability)* expr) => self.visit_record_access(expr),
                    &$($mutability)* Expr::RecordIndex(ref $($mutability)* expr) => self.visit_record_index(expr),
                    &$($mutability)* Expr::RecordUpdate(ref $($mutability)* expr) => self.visit_record_update(expr),
                    &$($mutability)* Expr::ListComprehension(ref $($mutability)* expr) => self.visit_list_comprehension(expr),
                    &$($mutability)* Expr::BinaryComprehension(ref $($mutability)* expr) => self.visit_binary_comprehension(expr),
                    &$($mutability)* Expr::Generator(ref $($mutability)* expr) => self.visit_generator(expr),
                    &$($mutability)* Expr::BinaryGenerator(ref $($mutability)* expr) => self.visit_binary_generator(expr),
                    &$($mutability)* Expr::Begin(ref $($mutability)* expr) => self.visit_begin(expr),
                    &$($mutability)* Expr::Apply(ref $($mutability)* expr) => self.visit_apply(expr),
                    &$($mutability)* Expr::Remote(ref $($mutability)* expr) => self.visit_remote(expr),
                    &$($mutability)* Expr::BinaryExpr(ref $($mutability)* expr) => self.visit_binary_expr(expr),
                    &$($mutability)* Expr::UnaryExpr(ref $($mutability)* expr) => self.visit_unary_expr(expr),
                    &$($mutability)* Expr::Match(ref $($mutability)* expr) => self.visit_match(expr),
                    &$($mutability)* Expr::If(ref $($mutability)* expr) => self.visit_if(expr),
                    &$($mutability)* Expr::Catch(ref $($mutability)* expr) => self.visit_catch(expr),
                    &$($mutability)* Expr::Case(ref $($mutability)* expr) => self.visit_case(expr),
                    &$($mutability)* Expr::Receive(ref $($mutability)* expr) => self.visit_receive(expr),
                    &$($mutability)* Expr::Try(ref $($mutability)* expr) => self.visit_try(expr),
                    &$($mutability)* Expr::Fun(ref $($mutability)* expr) => self.visit_fun(expr),
                }
            }

            fn visit_name(&mut self, name: &'ast $($mutability)* Name) {
                if let &$($mutability)* Name::Var(ref $($mutability)* var) = name {
                    self.visit_identifier(var);
                }
            }

            fn visit_identifier(&mut self, _identifier: &'ast $($mutability)* Ident) {}

            fn visit_symbol(&mut self, _symbol: &'ast $($mutability)* Symbol) {}

            fn visit_literal(&mut self, _literal: &'ast $($mutability)* Literal) {}

            fn visit_function_name(&mut self, name: &'ast $($mutability)* FunctionName) {
                self.walk_function_name(name);
            }

            fn visit_resolved_function_name(&mut self, _name: &'ast $($mutability)* ResolvedFunctionName) {}

            fn visit_partially_resolved_function_name(&mut self, _name: &'ast $($mutability)* PartiallyResolvedFunctionName) {}

            fn visit_unresolved_function_name(&mut self, name: &'ast $($mutability)* UnresolvedFunctionName) {
                self.walk_unresolved_function_name(name);
            }

            fn visit_nil(&mut self, _nil: &'ast $($mutability)* Nil) {}

            fn visit_cons(&mut self, cons: &'ast $($mutability)* Cons) {
                self.walk_cons(cons);
            }

            fn visit_tuple(&mut self, tuple: &'ast $($mutability)* Tuple) {
                self.walk_tuple(tuple);
            }

            fn visit_map(&mut self, map: &'ast $($mutability)* Map) {
                self.walk_map(map);
            }

            fn visit_map_field(&mut self, field: &'ast $($mutability)* MapField) {
                self.walk_map_field(field);
            }

            fn visit_map_update(&mut self, map: &'ast $($mutability)* MapUpdate) {
                self.walk_map_update(map);
            }

            fn visit_map_projection(&mut self, map: &'ast $($mutability)* MapProjection) {
                self.walk_map_projection(map);
            }

            fn visit_binary(&mut self, binary: &'ast $($mutability)* Binary) {
                self.walk_binary(binary);
            }

            fn visit_binary_element(&mut self, element: &'ast $($mutability)* BinaryElement) {
                self.walk_binary_element(element);
            }

            fn visit_bit_type(&mut self, _ty: &'ast $($mutability)* BitType) {}

            fn visit_record_access(&mut self, ra: &'ast $($mutability)* RecordAccess) {
                self.walk_record_access(ra);
            }

            fn visit_record_index(&mut self, _ri: &'ast $($mutability)* RecordIndex) {}

            fn visit_record_update(&mut self, ru: &'ast $($mutability)* RecordUpdate) {
                self.walk_record_update(ru);
            }

            fn visit_list_comprehension(&mut self, lc: &'ast $($mutability)* ListComprehension) {
                self.walk_list_comprehension(lc);
            }

            fn visit_binary_comprehension(&mut self, bc: &'ast $($mutability)* BinaryComprehension) {
                self.walk_binary_comprehension(bc);
            }

            fn visit_generator(&mut self, generator: &'ast $($mutability)* Generator) {
                self.walk_generator(generator);
            }

            fn visit_binary_generator(&mut self, generator: &'ast $($mutability)* BinaryGenerator) {
                self.walk_binary_generator(generator);
            }

            fn visit_begin(&mut self, begin: &'ast $($mutability)* Begin) {
                self.walk_begin(begin);
            }

            fn visit_apply(&mut self, apply: &'ast $($mutability)* Apply) {
                self.walk_apply(apply);
            }

            fn visit_remote(&mut self, remote: &'ast $($mutability)* Remote) {
                self.walk_remote(remote);
            }

            fn visit_binary_expr(&mut self, expr: &'ast $($mutability)* BinaryExpr) {
                self.walk_binary_expr(expr);
            }

            fn visit_unary_expr(&mut self, expr: &'ast $($mutability)* UnaryExpr) {
                self.walk_unary_expr(expr);
            }

            fn visit_match(&mut self, expr: &'ast $($mutability)* Match) {
                self.walk_match(expr);
            }

            fn visit_if(&mut self, expr: &'ast $($mutability)* If) {
                self.walk_if(expr);
            }

            fn visit_if_clause(&mut self, expr: &'ast $($mutability)* IfClause) {
                self.walk_if_clause(expr);
            }

            fn visit_catch(&mut self, expr: &'ast $($mutability)* Catch) {
                self.walk_catch(expr);
            }

            fn visit_case(&mut self, expr: &'ast $($mutability)* Case) {
                self.walk_case(expr);
            }

            fn visit_receive(&mut self, expr: &'ast $($mutability)* Receive) {
                self.walk_receive(expr);
            }

            fn visit_after(&mut self, expr: &'ast $($mutability)* After) {
                self.walk_after(expr);
            }

            fn visit_try(&mut self, expr: &'ast $($mutability)* Try) {
                self.walk_try(expr);
            }

            fn visit_try_clause(&mut self, expr: &'ast $($mutability)* TryClause) {
                self.walk_try_clause(expr);
            }

            fn visit_clause(&mut self, expr: &'ast $($mutability)* Clause) {
                self.walk_clause(expr);
            }

            fn visit_fun(&mut self, expr: &'ast $($mutability)* Function) {
                self.walk_fun(expr);
            }

            fn visit_lambda(&mut self, expr: &'ast $($mutability)* Lambda) {
                self.walk_lambda(expr);
            }

            fn visit_named_lambda(&mut self, expr: &'ast $($mutability)* NamedFunction) {
                self.walk_named_lambda(expr);
            }

            // The `walk` functions are not meant to be overridden.

            fn walk_module(&mut self, module: &'ast $($mutability)* Module) {
                self.visit_imports(&$($mutability)* module.imports);

                self.visit_exports(&$($mutability)* module.exports);

                for (_, type_def) in &$($mutability)* module.types {
                    self.visit_type_def(type_def);
                }

                self.visit_exported_types(&$($mutability)* module.exported_types);

                self.visit_behaviours(&$($mutability)* module.behaviours);

                for (_, callback) in &$($mutability)* module.callbacks {
                    self.visit_callback(callback);
                }

                for (_, record) in &$($mutability)* module.records {
                    self.visit_record(record);
                }

                for (_, attribute) in &$($mutability)* module.attributes {
                    self.visit_attribute(attribute);
                }

                for (_, function) in &$($mutability)* module.functions {
                    self.visit_function(function);
                }
            }

            fn walk_type_def(&mut self, type_def: &'ast $($mutability)* TypeDef) {
                self.visit_type(&$($mutability)* type_def.ty);
            }

            fn walk_record(&mut self, record: &'ast $($mutability)* Record) {
                for field in &$($mutability)* record.fields {
                    self.visit_record_field(field);
                }
            }

            fn walk_record_field(&mut self, field: &'ast $($mutability)* RecordField) {
                self.visit_name(&$($mutability)* field.name);
                if let Some(ref $($mutability)* expr) = field.value {
                    self.visit_expression(expr);
                }
                if let Some(ref $($mutability)* ty) = field.ty {
                    self.visit_type(ty);
                }
            }

            fn walk_callback(&mut self, callback: &'ast $($mutability)* Callback) {
                for sig in &$($mutability)* callback.sigs {
                    self.visit_callback_signature(sig);
                }
            }

            fn walk_spec(&mut self, spec: &'ast $($mutability)* TypeSpec) {
                for sig in &$($mutability)* spec.sigs {
                    self.visit_spec_signature(sig);
                }
            }

            fn walk_sig(&mut self, sig: &'ast $($mutability)* TypeSig) {
                for param in &$($mutability)* sig.params {
                    self.visit_type(param);
                }

                self.visit_type(&$($mutability)* sig.ret);

                match sig.guards {
                    None => (),
                    Some(ref $($mutability)* guards) => {
                        for guard in guards {
                            self.visit_type_guard(guard);
                        }
                    }
                }
            }

            fn walk_attribute(&mut self, attribute: &'ast $($mutability)* UserAttribute) {
                self.visit_expression(&$($mutability)* attribute.value);
            }

            fn walk_function(&mut self, function: &'ast $($mutability)* NamedFunction) {
                for clause in &$($mutability)* function.clauses {
                    self.visit_function_clause(clause);
                }
                match function.spec {
                    None => (),
                    Some(ref $($mutability)* spec) => self.visit_spec(spec)
                }
            }

            fn walk_function_clause(&mut self, clause: &'ast $($mutability)* FunctionClause) {
                for param in &$($mutability)* clause.params {
                    self.visit_expression(param);
                }

                match clause.guard {
                    None => (),
                    Some(ref $($mutability)* guards) => {
                        for guard in guards {
                            self.visit_guard(guard);
                        }
                    }
                }

                for expr in &$($mutability)* clause.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_guard(&mut self, guard: &'ast $($mutability)* Guard) {
                for condition in &$($mutability)* guard.conditions {
                    self.visit_expression(condition);
                }
            }

            fn walk_type(&mut self, _ty: &'ast $($mutability)* Type) {}

            fn walk_type_guard(&mut self, guard: &'ast $($mutability)* TypeGuard) {
                self.visit_identifier(&$($mutability)* guard.var);
                self.visit_type(&$($mutability)* guard.ty);
            }


            fn walk_function_name(&mut self, name: &'ast $($mutability)* FunctionName) {
                match name {
                    &$($mutability)* FunctionName::Resolved(ref $($mutability)* name) => {
                        self.visit_resolved_function_name(name)
                    }
                    &$($mutability)* FunctionName::PartiallyResolved(ref $($mutability)* name) => {
                        self.visit_partially_resolved_function_name(name)
                    }
                    &$($mutability)* FunctionName::Unresolved(ref $($mutability)* name) => {
                        self.visit_unresolved_function_name(name)
                    }
                }
            }

            fn walk_unresolved_function_name(&mut self, name: &'ast $($mutability)* UnresolvedFunctionName) {
                if let Some(ref $($mutability)* m) = name.module {
                    self.visit_name(m);
                }

                self.visit_name(&$($mutability)* name.function);
            }

            fn walk_cons(&mut self, cons: &'ast $($mutability)* Cons) {
                self.visit_expression(&$($mutability)* cons.head);
                self.visit_expression(&$($mutability)* cons.tail);
            }

            fn walk_tuple(&mut self, tuple: &'ast $($mutability)* Tuple) {
                for element in &$($mutability)* tuple.elements {
                    self.visit_expression(element);
                }
            }

            fn walk_map(&mut self, map: &'ast $($mutability)* Map) {
                for field in &$($mutability)* map.fields {
                    self.visit_map_field(field);
                }
            }

            fn walk_map_field(&mut self, field: &'ast $($mutability)* MapField) {
                match field {
                    &$($mutability)* MapField::Assoc { ref $($mutability)* key, ref $($mutability)* value, .. } => {
                        self.visit_expression(key);
                        self.visit_expression(value);
                    }
                    &$($mutability)* MapField::Exact { ref $($mutability)* key, ref $($mutability)* value, .. } => {
                        self.visit_expression(key);
                        self.visit_expression(value);
                    }
                }
            }

            fn walk_map_update(&mut self, map: &'ast $($mutability)* MapUpdate) {
                self.visit_expression(&$($mutability)* map.map);

                for update in &$($mutability)* map.updates {
                    self.visit_map_field(update);
                }
            }

            fn walk_map_projection(&mut self, map: &'ast $($mutability)* MapProjection) {
                self.visit_expression(&$($mutability)* map.map);

                for field in &$($mutability)* map.fields {
                    self.visit_map_field(field);
                }
            }

            fn walk_binary(&mut self, bin: &'ast $($mutability)* Binary) {
                for element in &$($mutability)* bin.elements {
                    self.visit_binary_element(element);
                }
            }

            fn walk_binary_element(&mut self, element: &'ast $($mutability)* BinaryElement) {
                self.visit_expression(&$($mutability)* element.bit_expr);

                if let Some(ref $($mutability)* expr) = element.bit_size {
                    self.visit_expression(expr);
                }

                if let Some(ref $($mutability)* types) = element.bit_type {
                    for ty in types {
                        self.visit_bit_type(ty);
                    }
                }
            }

            fn walk_record_access(&mut self, ra: &'ast $($mutability)* RecordAccess) {
                self.visit_expression(&$($mutability)* ra.record);
            }

            fn walk_record_update(&mut self, ru: &'ast $($mutability)* RecordUpdate) {
                self.visit_expression(&$($mutability)* ru.record);
                for update in &$($mutability)* ru.updates {
                    self.visit_record_field(update);
                }
            }

            fn walk_list_comprehension(&mut self, lc: &'ast $($mutability)* ListComprehension) {
                self.visit_expression(&$($mutability)* lc.body);
                for qualifier in &$($mutability)* lc.qualifiers {
                    self.visit_expression(qualifier);
                }
            }

            fn walk_binary_comprehension(&mut self, bc: &'ast $($mutability)* BinaryComprehension) {
                self.visit_expression(&$($mutability)* bc.body);
                for qualifier in &$($mutability)* bc.qualifiers {
                    self.visit_expression(qualifier);
                }
            }

            fn walk_generator(&mut self, generator: &'ast $($mutability)* Generator) {
                self.visit_expression(&$($mutability)* generator.pattern);
                self.visit_expression(&$($mutability)* generator.expr);
            }

            fn walk_binary_generator(&mut self, generator: &'ast $($mutability)* BinaryGenerator) {
                self.visit_expression(&$($mutability)* generator.pattern);
                self.visit_expression(&$($mutability)* generator.expr);
            }

            fn walk_begin(&mut self, begin: &'ast $($mutability)* Begin) {
                for expr in &$($mutability)* begin.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_apply(&mut self, apply: &'ast $($mutability)* Apply) {
                self.visit_expression(&$($mutability)* apply.callee);
                for arg in &$($mutability)* apply.args {
                    self.visit_expression(arg);
                }
            }

            fn walk_remote(&mut self, remote: &'ast $($mutability)* Remote) {
                self.visit_expression(&$($mutability)* remote.module);
                self.visit_expression(&$($mutability)* remote.function);
            }

            fn walk_binary_expr(&mut self, expr: &'ast $($mutability)* BinaryExpr) {
                self.visit_expression(&$($mutability)* expr.lhs);
                self.visit_expression(&$($mutability)* expr.rhs);
            }

            fn walk_unary_expr(&mut self, expr: &'ast $($mutability)* UnaryExpr) {
                self.visit_expression(&$($mutability)* expr.operand);
            }

            fn walk_match(&mut self, expr: &'ast $($mutability)* Match) {
                self.visit_expression(&$($mutability)* expr.pattern);
                self.visit_expression(&$($mutability)* expr.expr);
            }

            fn walk_if(&mut self, expr: &'ast $($mutability)* If) {
                for clause in &$($mutability)* expr.clauses {
                    self.visit_if_clause(clause);
                }
            }

            fn walk_if_clause(&mut self, clause: &'ast $($mutability)* IfClause) {
                for condition in &$($mutability)* clause.conditions {
                    self.visit_expression(condition);
                }
                for expr in &$($mutability)* clause.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_catch(&mut self, expr: &'ast $($mutability)* Catch) {
                self.visit_expression(&$($mutability)* expr.expr);
            }

            fn walk_case(&mut self, case: &'ast $($mutability)* Case) {
                self.visit_expression(&$($mutability)* case.expr);

                for clause in &$($mutability)* case.clauses {
                    self.visit_clause(clause);
                }
            }

            fn walk_receive(&mut self, receive: &'ast $($mutability)* Receive) {
                if let Some(ref $($mutability)* clauses) = receive.clauses {
                    for clause in clauses {
                        self.visit_clause(clause);
                    }
                }
                if let Some(ref $($mutability)* after) = receive.after {
                    self.visit_after(after);
                }
            }

            fn walk_after(&mut self, after: &'ast $($mutability)* After) {
                self.visit_expression(&$($mutability)* after.timeout);
                for expr in &$($mutability)* after.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_try(&mut self, try_expr: &'ast $($mutability)* Try) {
                if let Some(ref $($mutability)* exprs) = try_expr.exprs {
                    for ex in exprs {
                        self.visit_expression(ex);
                    }
                }
                if let Some(ref $($mutability)* clauses) = try_expr.clauses {
                    for clause in clauses {
                        self.visit_clause(clause);
                    }
                }
                if let Some(ref $($mutability)* catch_clauses) = try_expr.catch_clauses {
                    for catch_clause in catch_clauses {
                        self.visit_try_clause(catch_clause);
                    }
                }
                if let Some(ref $($mutability)* after) = try_expr.after {
                    for ex in after {
                        self.visit_expression(ex);
                    }
                }
            }

            fn walk_try_clause(&mut self, clause: &'ast $($mutability)* TryClause) {
                self.visit_name(&$($mutability)* clause.kind);
                self.visit_expression(&$($mutability)* clause.error);
                if let Some(ref $($mutability)* guards) = clause.guard {
                    for guard in guards {
                        self.visit_guard(guard);
                    }
                }
                self.visit_identifier(&$($mutability)* clause.trace);
                for expr in &$($mutability)* clause.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_clause(&mut self, clause: &'ast $($mutability)* Clause) {
                self.visit_expression(&$($mutability)* clause.pattern);
                if let Some(ref $($mutability)* guards) = clause.guard {
                    for guard in guards {
                        self.visit_guard(guard);
                    }
                }
                for expr in &$($mutability)* clause.body {
                    self.visit_expression(expr);
                }
            }

            fn walk_fun(&mut self, fun: &'ast $($mutability)* Function) {
                match fun {
                    Function::Named(ref $($mutability)* named) => self.visit_named_lambda(named),
                    Function::Unnamed(ref $($mutability)* lambda) => self.visit_lambda(lambda),
                }
            }

            fn walk_lambda(&mut self, lambda: &'ast $($mutability)* Lambda) {
                for clause in &$($mutability)* lambda.clauses {
                    self.visit_function_clause(clause);
                }
            }

            fn walk_named_lambda(&mut self, lambda: &'ast $($mutability)* NamedFunction) {
                for clause in &$($mutability)* lambda.clauses {
                    self.visit_function_clause(clause);
                }
            }
        }
    }
}

make_visitor!(ImmutableVisitor,);
make_visitor!(MutableVisitor, mut);
