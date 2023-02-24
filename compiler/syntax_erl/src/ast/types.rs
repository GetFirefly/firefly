use std::collections::HashSet;

use lazy_static::lazy_static;

use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_intern::{Ident, Symbol};
use firefly_number::Int;
use firefly_syntax_base::{BinaryOp, UnaryOp};

use crate::ast::Name;

lazy_static! {
    pub static ref BUILTIN_TYPES: HashSet<(Symbol, usize)> = {
        let mut bts = HashSet::new();
        bts.insert((Symbol::intern("any"), 0));
        bts.insert((Symbol::intern("arity"), 0));
        bts.insert((Symbol::intern("atom"), 0));
        bts.insert((Symbol::intern("binary"), 0));
        bts.insert((Symbol::intern("bitstring"), 0));
        bts.insert((Symbol::intern("bool"), 0));
        bts.insert((Symbol::intern("boolean"), 0));
        bts.insert((Symbol::intern("byte"), 0));
        bts.insert((Symbol::intern("char"), 0));
        bts.insert((Symbol::intern("float"), 0));
        bts.insert((Symbol::intern("function"), 0));
        bts.insert((Symbol::intern("identifier"), 0));
        bts.insert((Symbol::intern("integer"), 0));
        bts.insert((Symbol::intern("iodata"), 0));
        bts.insert((Symbol::intern("iolist"), 0));
        bts.insert((Symbol::intern("list"), 0));
        bts.insert((Symbol::intern("list"), 1));
        bts.insert((Symbol::intern("map"), 0));
        bts.insert((Symbol::intern("maybe_improper_list"), 0));
        bts.insert((Symbol::intern("maybe_improper_list"), 2));
        bts.insert((Symbol::intern("mfa"), 0));
        bts.insert((Symbol::intern("module"), 0));
        bts.insert((Symbol::intern("neg_integer"), 0));
        bts.insert((Symbol::intern("nil"), 0));
        bts.insert((Symbol::intern("no_return"), 0));
        bts.insert((Symbol::intern("node"), 0));
        bts.insert((Symbol::intern("non_neg_integer"), 0));
        bts.insert((Symbol::intern("none"), 0));
        bts.insert((Symbol::intern("nonempty_improper_list"), 2));
        bts.insert((Symbol::intern("nonempty_list"), 0));
        bts.insert((Symbol::intern("nonempty_list"), 1));
        bts.insert((Symbol::intern("nonempty_maybe_improper_list"), 0));
        bts.insert((Symbol::intern("nonempty_maybe_improper_list"), 2));
        bts.insert((Symbol::intern("nonempty_string"), 0));
        bts.insert((Symbol::intern("number"), 0));
        bts.insert((Symbol::intern("pid"), 0));
        bts.insert((Symbol::intern("port"), 0));
        bts.insert((Symbol::intern("pos_integer"), 0));
        bts.insert((Symbol::intern("reference"), 0));
        bts.insert((Symbol::intern("string"), 0));
        bts.insert((Symbol::intern("term"), 0));
        bts.insert((Symbol::intern("timeout"), 0));
        bts.insert((Symbol::intern("tuple"), 0));
        bts
    };
}

#[derive(Debug, Clone, Spanned)]
pub enum Type {
    Name(Name),
    Annotated {
        #[span]
        span: SourceSpan,
        name: Name,
        ty: Box<Type>,
    },
    Union {
        #[span]
        span: SourceSpan,
        types: Vec<Type>,
    },
    Range {
        #[span]
        span: SourceSpan,
        start: Box<Type>,
        end: Box<Type>,
    },
    BinaryOp {
        #[span]
        span: SourceSpan,
        lhs: Box<Type>,
        op: BinaryOp,
        rhs: Box<Type>,
    },
    UnaryOp {
        #[span]
        span: SourceSpan,
        op: UnaryOp,
        rhs: Box<Type>,
    },
    Generic {
        #[span]
        span: SourceSpan,
        fun: Ident,
        params: Vec<Type>,
    },
    Remote {
        #[span]
        span: SourceSpan,
        module: Ident,
        fun: Ident,
        args: Vec<Type>,
    },
    Nil(SourceSpan),
    List(#[span] SourceSpan, Box<Type>),
    NonEmptyList(#[span] SourceSpan, Box<Type>),
    Map(#[span] SourceSpan, Vec<Type>),
    Tuple(#[span] SourceSpan, Vec<Type>),
    Record(#[span] SourceSpan, Ident, Vec<Type>),
    Binary(#[span] SourceSpan, Box<Type>, Box<Type>),
    Integer(#[span] SourceSpan, Int),
    Char(#[span] SourceSpan, char),
    AnyFun {
        #[span]
        span: SourceSpan,
        ret: Option<Box<Type>>,
    },
    Fun {
        #[span]
        span: SourceSpan,
        params: Vec<Type>,
        ret: Box<Type>,
    },
    KeyValuePair(#[span] SourceSpan, Box<Type>, Box<Type>),
    Field(#[span] SourceSpan, Ident, Box<Type>),
}
impl Type {
    pub fn union(span: SourceSpan, lhs: Type, rhs: Type) -> Self {
        let mut types = match lhs {
            Type::Union { types, .. } => types,
            ty => vec![ty],
        };
        let mut rest = match rhs {
            Type::Union { types, .. } => types,
            ty => vec![ty],
        };
        types.append(&mut rest);
        Type::Union { span, types }
    }

    pub fn is_builtin_type(&self) -> bool {
        match self {
            &Type::Name(Name::Atom(Ident { ref name, .. })) => BUILTIN_TYPES.contains(&(*name, 0)),
            &Type::Annotated { ref name, .. } => match name {
                Name::Atom(v) => BUILTIN_TYPES.contains(&(v.name, 0)),
                Name::Var(v) => BUILTIN_TYPES.contains(&(v.name, 0)),
            },
            &Type::Generic {
                ref fun,
                ref params,
                ..
            } => BUILTIN_TYPES.contains(&(fun.name, params.len())),
            &Type::Nil(_) => true,
            &Type::List(_, _) => true,
            &Type::NonEmptyList(_, _) => true,
            &Type::Map(_, _) => true,
            &Type::Tuple(_, _) => true,
            &Type::Record(_, _, _) => true,
            &Type::Binary(_, _, _) => true,
            &Type::Integer(_, _) => true,
            &Type::Char(_, _) => true,
            &Type::AnyFun { .. } => true,
            &Type::Fun { .. } => true,
            &Type::KeyValuePair(_, _, _) => true,
            &Type::Field(_, _, _) => true,
            _ => false,
        }
    }
}
impl PartialEq for Type {
    fn eq(&self, other: &Type) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (
                Type::Annotated {
                    name: ref x1,
                    ty: ref x2,
                    ..
                },
                Type::Annotated {
                    name: ref y1,
                    ty: ref y2,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2),
            (
                Type::Union {
                    types: ref types1, ..
                },
                Type::Union {
                    types: ref types2, ..
                },
            ) => types1 == types2,
            (
                Type::Range {
                    start: ref x1,
                    end: ref x2,
                    ..
                },
                Type::Range {
                    start: ref y1,
                    end: ref y2,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2),
            (
                Type::BinaryOp {
                    lhs: ref x1,
                    op: ref x2,
                    rhs: ref x3,
                    ..
                },
                Type::BinaryOp {
                    lhs: ref y1,
                    op: ref y2,
                    rhs: ref y3,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2) && (x3 == y3),
            (
                Type::UnaryOp {
                    op: ref x1,
                    rhs: ref x2,
                    ..
                },
                Type::UnaryOp {
                    op: ref y1,
                    rhs: ref y2,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2),
            (
                Type::Generic {
                    fun: ref x1,
                    params: ref x2,
                    ..
                },
                Type::Generic {
                    fun: ref y1,
                    params: ref y2,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2),
            (
                Type::Remote {
                    module: ref x1,
                    fun: ref x2,
                    args: ref x3,
                    ..
                },
                Type::Remote {
                    module: ref y1,
                    fun: ref y2,
                    args: ref y3,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2) && (x3 == y3),
            (Type::Nil(_), Type::Nil(_)) => true,
            (Type::List(_, ref x), Type::List(_, ref y)) => x == y,
            (Type::NonEmptyList(_, ref x), Type::List(_, ref y)) => x == y,
            (Type::Map(_, ref x), Type::Map(_, ref y)) => x == y,
            (Type::Tuple(_, ref x), Type::Tuple(_, ref y)) => x == y,
            (Type::Record(_, ref x1, ref x2), Type::Record(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (Type::Binary(_, ref m1, ref n1), Type::Binary(_, ref m2, ref n2)) => {
                (m1 == m2) && (n1 == n2)
            }
            (Type::Integer(_, x), Type::Integer(_, y)) => x == y,
            (Type::Char(_, x), Type::Char(_, y)) => x == y,
            (Type::AnyFun { ret: x, .. }, Type::AnyFun { ret: y, .. }) => x == y,
            (
                Type::Fun {
                    params: ref x1,
                    ret: ref x2,
                    ..
                },
                Type::Fun {
                    params: ref y1,
                    ret: ref y2,
                    ..
                },
            ) => (x1 == y1) && (x2 == y2),
            (Type::KeyValuePair(_, ref x1, ref x2), Type::KeyValuePair(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (Type::Field(_, ref x1, ref x2), Type::Field(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            _ => false,
        }
    }
}
