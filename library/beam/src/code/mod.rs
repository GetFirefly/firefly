pub mod ast;

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::path::Path;

use firefly_intern::Symbol;

use crate::reader::RawBeamFile;
use crate::serialization::etf;
use crate::serialization::etf::pattern;
use crate::serialization::etf::pattern::{Ascii, Unicode};
use crate::serialization::etf::pattern::{FixList, VarList};
use crate::serialization::etf::pattern::{Int, F64, U32, U64, U8};
use crate::serialization::etf::pattern::{MatchResult, Pattern, Unmatch};
use crate::serialization::etf::pattern::{Or, Union2, Union3};
use crate::FromBeamError;

use self::ast::*;

macro_rules! to {
    ($to:ty) => {
        To::<$to>(PhantomData)
    };
}

macro_rules! return_if_ok {
    ($e:expr) => {
        match $e {
            Ok(value) => return Ok(From::from(value)),
            Err(err) => err,
        }
    };
}

/// This type represents the Abstract Erlang syntax tree contained in the Abst chunk of a BEAM file
#[derive(Debug)]
pub struct AbstractCode {
    pub forms: Vec<Form>,
}
impl AbstractCode {
    pub fn from_beam_file<P: AsRef<Path>>(path: P) -> Result<Self, FromBeamError> {
        use crate::serialization::etf::Term;

        let beam = RawBeamFile::from_file(path)?;

        // Try to get the Abst chunk first, but if it doesn't exist, use the Dbgi chunk
        let abst = beam.get_chunk(b"Abst");

        if let Some(chunk) = abst {
            let code = Term::decode(std::io::Cursor::new(&chunk.data))?;
            let (_, forms) = code
                .as_match(("raw_abstract_v1", VarList(to!(Form))))
                .map_err(|_| FromBeamError::NoDebugInfo)?;

            return Ok(AbstractCode { forms });
        }

        let dbgi = beam.get_chunk(b"Dbgi").ok_or(FromBeamError::NoDebugInfo)?;

        let debug_info = Term::decode(std::io::Cursor::new(&dbgi.data))?;
        let (_, _, (forms, _opts)) = debug_info
            .as_match((
                "debug_info_v1",
                "erl_abstract_code",
                (VarList(to!(Form)), pattern::any::<Term>()),
            ))
            .map_err(|err| {
                dbg!(&err.cause);
                FromBeamError::NoDebugInfo
            })?;

        Ok(AbstractCode { forms })
    }
}

trait FromTerm<'a> {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>>
    where
        Self: Sized;
}

struct To<T>(PhantomData<T>);
impl<T> Clone for To<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        To(PhantomData)
    }
}
impl<T> fmt::Debug for To<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "To<{}>", core::any::type_name::<T>())
    }
}

impl<'a, F> Pattern<'a> for To<F>
where
    F: FromTerm<'a> + Debug + 'static,
{
    type Output = F;
    fn try_match(&self, term: &'a etf::Term) -> MatchResult<'a, Self::Output> {
        F::try_from(term).map_err(|e| self.unmatched(term).cause(e))
    }
}

#[inline]
const fn any() -> etf::pattern::Any<etf::Term> {
    etf::pattern::any()
}

#[inline]
const fn atom() -> etf::pattern::AtomName {
    etf::pattern::AtomName
}

const fn expr() -> To<Expression> {
    to!(Expression)
}

const fn clause() -> To<Clause> {
    to!(Clause)
}

const fn ty() -> To<Type> {
    to!(Type)
}

const fn ftype() -> To<FunType> {
    to!(FunType)
}

const fn var() -> To<Var> {
    to!(Var)
}

const fn atom_lit() -> To<Atom> {
    to!(Atom)
}

const fn integer_lit() -> To<Integer> {
    to!(Integer)
}

impl<'a> FromTerm<'a> for Form {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        let e = return_if_ok!(term.as_match(to!(ModuleAttr)));
        let e = return_if_ok!(term.as_match(to!(ModuleAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Function))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(FileAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(BehaviourAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(ExportAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(ImportAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(ExportTypeAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(CompileOptionsAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RecordDef))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(TypeDef))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(SpecAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(CallbackAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(OnLoadAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(NifsAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(UserAttr))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Warning))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Eof))).max_depth(e);
        Err(e)
    }
}

impl<'a> FromTerm<'a> for Expression {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        let e = return_if_ok!(term.as_match(integer_lit()));
        let e = return_if_ok!(term.as_match(to!(Float))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Str))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Char))).max_depth(e);
        let e = return_if_ok!(term.as_match(atom_lit())).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Match))).max_depth(e);
        let e = return_if_ok!(term.as_match(var())).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Tuple))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Nil))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Cons))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Binary))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(UnaryOp))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(BinaryOp))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Record))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RecordIndex))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RecordAccess))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Remote))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Map))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Catch))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Call))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Comprehension))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Block))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(If))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Case))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Try))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Receive))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(InternalFun))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(ExternalFun))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(AnonymousFun))).max_depth(e);
        Err(e)
    }
}

impl<'a> FromTerm<'a> for Type {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        let e = return_if_ok!(term.as_match(integer_lit()));
        let e = return_if_ok!(term.as_match(atom_lit())).max_depth(e);
        let e = return_if_ok!(term.as_match(var())).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(UnaryTypeOp))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(BinaryTypeOp))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(Nil))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(AnnotatedType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(BitStringType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(AnyFunType))).max_depth(e);
        let e = return_if_ok!(term.as_match(ftype())).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RangeType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(MapType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RecordType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(RemoteType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(AnyTupleType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(TupleType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(UnionType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(ProductType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(BuiltInType))).max_depth(e);
        let e = return_if_ok!(term.as_match(to!(UserType))).max_depth(e);
        Err(e)
    }
}

impl<'a> FromTerm<'a> for Catch {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("catch", pattern::Location, expr()))
            .map(|(_, loc, expr)| Self::new(loc, expr))
    }
}

impl<'a> FromTerm<'a> for Receive {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            ("receive", pattern::Location, VarList(clause())),
            (
                "receive",
                pattern::Location,
                VarList(clause()),
                expr(),
                VarList(expr()),
            ),
        )))
        .map(|result| match result {
            Union2::A((_, loc, clauses)) => Self::new(loc, clauses, None, vec![]),
            Union2::B((_, loc, clauses, timeout, after)) => {
                Self::new(loc, clauses, Some(timeout), after)
            }
        })
    }
}

impl<'a> FromTerm<'a> for InternalFun {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("fun", pattern::Location, ("function", atom(), U8)))
            .map(|(_, loc, (_, name, arity))| Self::new(loc, name, arity))
    }
}

impl<'a> FromTerm<'a> for ExternalFun {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "fun",
            pattern::Location,
            ("function", expr(), expr(), expr()),
        ))
        .map(|(_, loc, (_, module, function, arity))| Self::new(loc, module, function, arity))
    }
}

impl<'a> FromTerm<'a> for AnonymousFun {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            ("fun", pattern::Location, ("clauses", VarList(clause()))),
            ("named_fun", pattern::Location, atom(), VarList(clause())),
        )))
        .map(|result| match result {
            Union2::A((_, loc, (_, clauses))) => Self::new(loc, None, clauses),
            Union2::B((_, loc, name, clauses)) => Self::new(loc, Some(name), clauses),
        })
    }
}

impl<'a> FromTerm<'a> for Block {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("block", pattern::Location, VarList(expr())))
            .map(|(_, loc, body)| Self::new(loc, body))
    }
}

impl<'a> FromTerm<'a> for If {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("if", pattern::Location, VarList(clause())))
            .map(|(_, loc, clauses)| Self::new(loc, clauses))
    }
}

impl<'a> FromTerm<'a> for Case {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("case", pattern::Location, expr(), VarList(clause())))
            .map(|(_, loc, expr, clauses)| Self::new(loc, expr, clauses))
    }
}

impl<'a> FromTerm<'a> for Try {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "try",
            pattern::Location,
            VarList(expr()),
            VarList(clause()),
            VarList(clause()),
            VarList(expr()),
        ))
        .map(|(_, loc, body, case_clauses, catch_clauses, after)| {
            Self::new(loc, body, case_clauses, catch_clauses, after)
        })
    }
}

impl<'a> FromTerm<'a> for Comprehension {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            Or(("lc", "bc")),
            pattern::Location,
            expr(),
            VarList(to!(Qualifier)),
        ))
        .map(|(is_lc, loc, expr, qualifiers)| Self::new(loc, is_lc.is_a(), expr, qualifiers))
    }
}

impl<'a> FromTerm<'a> for Qualifier {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            ("generate", pattern::Location, expr(), expr()),
            ("b_generate", pattern::Location, expr(), expr()),
            expr(),
        )))
        .map(|result| match result {
            Union3::A((_, loc, pattern, expr)) => {
                Qualifier::Generator(Generator::new(loc, pattern, expr))
            }
            Union3::B((_, loc, pattern, expr)) => {
                Qualifier::BitStringGenerator(Generator::new(loc, pattern, expr))
            }
            Union3::C(expr) => Qualifier::Filter(expr),
        })
    }
}

impl<'a> FromTerm<'a> for Clause {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "clause",
            pattern::Location,
            VarList(expr()),
            VarList(to!(OrGuard)),
            VarList(expr()),
        ))
        .map(|(_, loc, patterns, guards, body)| Self::new(loc, patterns, guards, body))
    }
}

impl<'a> FromTerm<'a> for OrGuard {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(VarList(expr()))
            .map(|guards| Self::new(guards))
    }
}

impl<'a> FromTerm<'a> for Tuple {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("tuple", pattern::Location, VarList(expr())))
            .map(|(_, loc, elements)| Self::new(loc, elements))
    }
}

impl<'a> FromTerm<'a> for Cons {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("cons", pattern::Location, expr(), expr()))
            .map(|(_, loc, head, tail)| Self::new(loc, head, tail))
    }
}

impl<'a> FromTerm<'a> for Binary {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("bin", pattern::Location, VarList(to!(BinElement))))
            .map(|(_, loc, elements)| Self::new(loc, elements))
    }
}

impl<'a> FromTerm<'a> for BinElement {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "bin_element",
            pattern::Location,
            expr(),
            Or((expr(), "default")),
            Or((VarList(to!(BinElementTypeSpec)), "default")),
        ))
        .map(|(_, loc, value, size, tsl)| {
            let size = if let Union2::A(size) = size {
                Some(size)
            } else {
                None
            };
            let tsl = if let Union2::A(tsl) = tsl {
                Some(tsl)
            } else {
                None
            };
            Self::new(loc, value, size, tsl)
        })
    }
}

impl<'a> FromTerm<'a> for BinElementTypeSpec {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((atom(), (atom(), U64))))
            .map(|ts| match ts {
                Union2::A(name) => Self::new(name, None),
                Union2::B((name, value)) => Self::new(name, Some(value)),
            })
    }
}

impl<'a> FromTerm<'a> for Record {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            (
                "record",
                pattern::Location,
                atom(),
                VarList(to!(RecordField)),
            ),
            (
                "record",
                pattern::Location,
                expr(),
                atom(),
                VarList(to!(RecordField)),
            ),
        )))
        .map(|result| match result {
            Union2::A((_, loc, name, fields)) => Self::new(loc, None, name, fields),
            Union2::B((_, loc, base, name, fields)) => Self::new(loc, Some(base), name, fields),
        })
    }
}

impl<'a> FromTerm<'a> for RecordField {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "record_field",
            pattern::Location,
            Or((atom_lit(), ("var", pattern::Location, "_"))),
            expr(),
        ))
        .map(|(_, loc, name, value)| {
            let name = name.into_result().ok().map(|n| n.value);
            Self::new(loc, name, value)
        })
    }
}

impl<'a> FromTerm<'a> for Remote {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("remote", pattern::Location, expr(), expr()))
            .map(|(_, loc, module, function)| Self::new(loc, module, function))
    }
}

impl<'a> FromTerm<'a> for Map {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            ("map", pattern::Location, VarList(to!(MapPair))),
            ("map", pattern::Location, expr(), VarList(to!(MapPair))),
        )))
        .map(|result| match result {
            Union2::A((_, loc, pairs)) => Self::new(loc, None, pairs),
            Union2::B((_, loc, base, pairs)) => Self::new(loc, Some(base), pairs),
        })
    }
}

impl<'a> FromTerm<'a> for MapPair {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            Or(("map_field_assoc", "map_field_exact")),
            pattern::Location,
            expr(),
            expr(),
        ))
        .map(|(is_assoc, loc, key, value)| Self::new(loc, is_assoc.is_a(), key, value))
    }
}

impl<'a> FromTerm<'a> for Call {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("call", pattern::Location, expr(), VarList(expr())))
            .map(|(_, loc, callee, args)| Self::new(loc, callee, args))
    }
}

impl<'a> FromTerm<'a> for RecordIndex {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("record_index", pattern::Location, atom(), atom_lit()))
            .map(|(_, loc, name, field)| Self::new(loc, name, field.value))
    }
}

impl<'a> FromTerm<'a> for RecordAccess {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "record_field",
            pattern::Location,
            expr(),
            atom(),
            atom_lit(),
        ))
        .map(|(_, loc, base, name, field)| Self::new(loc, base, name, field.value))
    }
}

impl<'a> FromTerm<'a> for BinaryOp {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("op", pattern::Location, atom(), expr(), expr()))
            .map(|(_, loc, op, left, right)| Self::new(loc, op, left, right))
    }
}

impl<'a> FromTerm<'a> for BinaryTypeOp {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("op", pattern::Location, atom(), expr(), expr()))
            .map(|(_, loc, op, left, right)| Self::new(loc, op, left, right))
    }
}

impl<'a> FromTerm<'a> for UnaryOp {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("op", pattern::Location, atom(), expr()))
            .map(|(_, loc, op, arg)| Self::new(loc, op, arg))
    }
}

impl<'a> FromTerm<'a> for UnaryTypeOp {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("op", pattern::Location, atom(), expr()))
            .map(|(_, loc, op, arg)| Self::new(loc, op, arg))
    }
}

impl<'a> FromTerm<'a> for Nil {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("nil", pattern::Location))
            .map(|(_, loc)| Self::new(loc))
    }
}

impl<'a> FromTerm<'a> for Match {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("match", pattern::Location, expr(), expr()))
            .map(|(_, loc, left, right)| Self::new(loc, left, right))
    }
}

impl<'a> FromTerm<'a> for UserType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("user_type", pattern::Location, atom(), VarList(ty())))
            .map(|(_, loc, name, args)| Self::new(loc, name, args))
    }
}

impl<'a> FromTerm<'a> for UnionType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "union", VarList(ty())))
            .map(|(_, loc, _, types)| Self::new(loc, types))
    }
}

impl<'a> FromTerm<'a> for ProductType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "product", VarList(ty())))
            .map(|(_, loc, _, types)| Self::new(loc, types))
    }
}

impl<'a> FromTerm<'a> for TupleType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "tuple", VarList(ty())))
            .map(|(_, loc, _, types)| Self::new(loc, types))
    }
}

impl<'a> FromTerm<'a> for AnyTupleType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "tuple", "any"))
            .map(|(_, loc, _, _)| Self::new(loc))
    }
}

impl<'a> FromTerm<'a> for RemoteType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "remote_type",
            pattern::Location,
            FixList((atom_lit(), atom_lit(), VarList(ty()))),
        ))
        .map(|(_, loc, (module, function, args))| {
            Self::new(
                loc,
                FunctionName {
                    module: Some(module.value),
                    name: function.value,
                    arity: args.len().try_into().unwrap(),
                },
                args,
            )
        })
    }
}

impl<'a> FromTerm<'a> for RecordType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "record",
            pattern::Cons(atom_lit(), to!(RecordFieldType)),
        ))
        .map(|(_, loc, _, (name, fields))| Self::new(loc, name.value, fields))
    }
}

impl<'a> FromTerm<'a> for RecordFieldType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "field_type",
            FixList((atom_lit(), ty())),
        ))
        .map(|(_, loc, _, (name, ty))| Self::new(loc, name.value, ty))
    }
}

impl<'a> FromTerm<'a> for BuiltInType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            atom(),
            Or(("any", VarList(ty()))),
        ))
        .map(|(_, loc, name, args)| match args {
            Union2::A(_) => Self::new(loc, name, Vec::new()),
            Union2::B(args) => Self::new(loc, name, args),
        })
    }
}

impl<'a> FromTerm<'a> for MapType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "map", VarList(to!(MapPairType))))
            .map(|(_, loc, _, pairs)| Self::new(loc, pairs))
    }
}

impl<'a> FromTerm<'a> for MapPairType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "map_field_assoc",
            FixList((ty(), ty())),
        ))
        .map(|(_, loc, _, (key, value))| Self::new(loc, key, value))
    }
}

impl<'a> FromTerm<'a> for RangeType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("type", pattern::Location, "range", FixList((ty(), ty()))))
            .map(|(_, loc, _, (low, high))| Self::new(loc, low, high))
    }
}

impl<'a> FromTerm<'a> for FunType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            (
                "type",
                pattern::Location,
                "bounded_fun",
                FixList((ftype(), VarList(to!(Constraint)))),
            ),
            (
                "type",
                pattern::Location,
                "fun",
                FixList((("type", pattern::Location, "product", VarList(ty())), ty())),
            ),
        )))
        .map(|result| match result {
            Union2::A((_, _, _, (mut fun, constraints))) => {
                fun.constraints = constraints;
                fun
            }
            Union2::B((_, loc, _, ((_, _, _, args), return_type))) => {
                Self::new(loc, args, return_type)
            }
        })
    }
}

impl<'a> FromTerm<'a> for Constraint {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "constraint",
            FixList((
                ("atom", pattern::Location, "is_subtype"),
                FixList((var(), ty())),
            )),
        ))
        .map(|(_, loc, _, (_, (var, subtype)))| Self::new(loc, var, subtype))
    }
}

impl<'a> FromTerm<'a> for AnyFunType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "fun",
            Or((
                pattern::Nil,
                FixList((("type", pattern::Location, "any"), ty())),
            )),
        ))
        .map(|(_, loc, _, fun)| match fun {
            Union2::A(_) => Self::new(loc, None),
            Union2::B((_, ty)) => Self::new(loc, Some(ty)),
        })
    }
}

impl<'a> FromTerm<'a> for AnnotatedType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("ann_type", pattern::Location, FixList((var(), ty()))))
            .map(|(_, loc, (var, ty))| Self::new(loc, var, ty))
    }
}

impl<'a> FromTerm<'a> for BitStringType {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "type",
            pattern::Location,
            "binary",
            FixList((integer_lit(), integer_lit())),
        ))
        .map(|(_, loc, _, (bytes, bits))| {
            Self::new(loc, bytes.to_u64().unwrap(), bits.to_u64().unwrap())
        })
    }
}

impl<'a> FromTerm<'a> for ModuleAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("attribute", pattern::Location, "module", atom()))
            .map(|(_, loc, _, name)| Self::new(loc, name))
    }
}

impl<'a> FromTerm<'a> for FileAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "file",
            (pattern::Str(Ascii), U32),
        ))
        .map(|(_, loc, _, (original_file, original_line))| {
            Self::new(loc, Symbol::intern(&original_file), original_line)
        })
    }
}

impl<'a> FromTerm<'a> for BehaviourAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            Or(("behaviour", "behavior")),
            atom(),
        ))
        .map(|(_, loc, _, name)| Self::new(loc, name))
    }
}

impl<'a> FromTerm<'a> for RecordDef {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "record",
            (atom(), VarList(to!(RecordFieldDef))),
        ))
        .map(|(_, loc, _, (name, fields))| Self::new(loc, name, fields))
    }
}

impl<'a> FromTerm<'a> for RecordFieldDef {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(Or((
            ("record_field", pattern::Location, atom_lit()),
            ("record_field", pattern::Location, atom_lit(), expr()),
            ("typed_record_field", to!(RecordFieldDef), ty()),
        )))
        .map(|result| match result {
            Union3::A((_, loc, name)) => Self::new(loc, name.value, Type::any(loc), None),
            Union3::B((_, loc, name, value)) => {
                Self::new(loc, name.value, Type::any(loc), Some(value))
            }
            Union3::C((_, mut field, ty)) => {
                field.ty = ty;
                field
            }
        })
    }
}

impl<'a> FromTerm<'a> for Atom {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("atom", pattern::Location, atom()))
            .map(|(_, loc, name)| Self::new(loc, name))
    }
}

impl<'a> FromTerm<'a> for Integer {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("integer", pattern::Location, Int))
            .map(|(_, loc, value)| Self::new(loc, value.into()))
    }
}

impl<'a> FromTerm<'a> for Char {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("char", pattern::Location, Unicode))
            .map(|(_, loc, ch)| Self::new(loc, ch))
    }
}

impl<'a> FromTerm<'a> for Float {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("float", pattern::Location, F64))
            .map(|(_, loc, value)| Self::new(loc, value.into()))
    }
}

impl<'a> FromTerm<'a> for Str {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("string", pattern::Location, pattern::Str(Unicode)))
            .map(|(_, loc, value)| Self::new(loc, Symbol::intern(&value)))
    }
}

impl<'a> FromTerm<'a> for Var {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("var", pattern::Location, atom()))
            .map(|(_, loc, name)| Self::new(loc, name))
    }
}

impl<'a> FromTerm<'a> for TypeDef {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            Or(("opaque", "type")),
            (atom(), any(), VarList(var())),
        ))
        .map(|(_, loc, kind, (name, maybe_ty, vars))| {
            let ty = match maybe_ty.as_match(ty()) {
                Ok(ty) => ty,
                Err(err) => panic!("unrecognized type construct: {:#?}", &err),
            };
            let mut def = Self::new(loc, name, vars, ty);
            def.is_opaque = kind.is_a();
            def
        })
    }
}

impl<'a> FromTerm<'a> for Function {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("function", pattern::Location, atom(), U8, VarList(clause())))
            .map(|(_, loc, name, arity, clauses)| {
                Self::new(
                    loc,
                    FunctionName {
                        module: None,
                        name,
                        arity,
                    },
                    clauses,
                )
            })
    }
}

impl<'a> FromTerm<'a> for SpecAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "spec",
            (pattern::FunctionName, VarList(ty())),
        ))
        .map(|(_, loc, _, (name, clauses))| Self { loc, name, clauses })
    }
}

impl<'a> FromTerm<'a> for CallbackAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "callback",
            (pattern::FunctionName, VarList(ty())),
        ))
        .map(|(_, loc, _, (name, clauses))| Self { loc, name, clauses })
    }
}

impl<'a> FromTerm<'a> for ExportAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "export",
            VarList(pattern::FunctionName),
        ))
        .map(|(_, loc, _, functions)| Self::new(loc, functions))
    }
}

impl<'a> FromTerm<'a> for NifsAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "nifs",
            VarList(pattern::FunctionName),
        ))
        .map(|(_, loc, _, funs)| Self { loc, funs })
    }
}

impl<'a> FromTerm<'a> for OnLoadAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "on_load",
            pattern::FunctionName,
        ))
        .map(|(_, loc, _, fun)| Self { loc, fun })
    }
}

impl<'a> FromTerm<'a> for ImportAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "import",
            (atom(), VarList(pattern::FunctionName)),
        ))
        .map(|(_, loc, _, (module, functions))| Self::new(loc, module, functions))
    }
}

impl<'a> FromTerm<'a> for ExportTypeAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match((
            "attribute",
            pattern::Location,
            "export_type",
            VarList(pattern::FunctionName),
        ))
        .map(|(_, loc, _, export_types)| Self::new(loc, export_types))
    }
}

impl<'a> FromTerm<'a> for CompileOptionsAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("attribute", pattern::Location, "compile", any()))
            .map(|(_, loc, _, opt_or_options)| match opt_or_options.clone() {
                etf::Term::List(list) => Self::new(loc, list.elements),
                term => Self::new(loc, vec![term]),
            })
    }
}

impl<'a> FromTerm<'a> for UserAttr {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("attribute", pattern::Location, atom(), any()))
            .map(|(_, loc, name, value)| Self::new(loc, name, value.clone()))
    }
}

impl<'a> FromTerm<'a> for Warning {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("warning", (pattern::Location, "epp", ("warning", any()))))
            .map(|(_, (loc, _, (_, message)))| Self {
                loc,
                message: message.clone(),
            })
    }
}

impl<'a> FromTerm<'a> for Eof {
    fn try_from(term: &'a etf::Term) -> Result<Self, Unmatch<'a>> {
        term.as_match(("eof", pattern::Location))
            .map(|(_, loc)| Self::new(loc))
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn decode_ast() {
        assert_matches!(AbstractCode::from_beam_file(test_file("test.beam")), Ok(_));
    }

    fn test_file(name: &str) -> PathBuf {
        let mut path = PathBuf::from("tests/testdata/ast");
        path.push(name);
        path
    }
}
