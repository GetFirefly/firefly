use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use firefly_binary::{BinaryEntrySpecifier, BitVec, Bitstring};
use firefly_diagnostics::{SourceSpan, Span, Spanned};
use firefly_intern::{symbols, Ident, Symbol};
use firefly_number::{Float, Int, Number};
use firefly_syntax_base::{self as syntax_base, Annotations, BinaryOp, FunctionName, UnaryOp};

use super::{Arity, Fun, FunctionVar, Guard, Name, Type};

use crate::evaluator::{self, EvalError};
use crate::lexer::DelayedSubstitution;

/// The set of all possible expressions
#[derive(Debug, Clone, PartialEq, Spanned)]
pub enum Expr {
    // An identifier/variable/function reference
    Var(Var),
    // Literal values
    Literal(Literal),
    FunctionVar(FunctionVar),
    // Delayed substitution of macro
    DelayedSubstitution(Span<DelayedSubstitution>),
    // The various list forms
    Cons(Cons),
    // Other data structures
    Tuple(Tuple),
    Map(Map),
    MapUpdate(MapUpdate),
    Binary(Binary),
    Record(Record),
    RecordAccess(RecordAccess),
    RecordIndex(RecordIndex),
    RecordUpdate(RecordUpdate),
    // Comprehensions
    ListComprehension(ListComprehension),
    BinaryComprehension(BinaryComprehension),
    Generator(Generator),
    // Complex expressions
    Begin(Begin),
    Apply(Apply),
    Remote(Remote),
    BinaryExpr(BinaryExpr),
    UnaryExpr(UnaryExpr),
    Match(Match),
    If(If),
    Catch(Catch),
    Case(Case),
    Receive(Receive),
    Try(Try),
    Fun(Fun),
    Protect(Protect),
}
impl Expr {
    pub fn try_resolve_apply(span: SourceSpan, callee: Expr, args: Vec<Expr>) -> Self {
        let arity = args.len().try_into().unwrap();
        match callee {
            Expr::Remote(remote) => match (remote.module.as_ref(), remote.function.as_ref()) {
                (Expr::Literal(Literal::Atom(m)), Expr::Literal(Literal::Atom(f))) => {
                    let name = FunctionVar::new(remote.span, m.name, f.name, arity);
                    Expr::Apply(Apply {
                        span,
                        callee: Box::new(Expr::FunctionVar(name)),
                        args,
                    })
                }
                _ => Expr::Apply(Apply {
                    span,
                    callee: Box::new(Expr::Remote(remote)),
                    args,
                }),
            },
            callee => Expr::Apply(Apply {
                span,
                callee: Box::new(callee),
                args,
            }),
        }
    }

    pub fn is_safe(&self) -> bool {
        match self {
            Self::Var(_) | Self::Literal(_) | Self::Cons(_) | Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub fn is_generator(&self) -> bool {
        match self {
            Self::Generator(_) => true,
            _ => false,
        }
    }

    /// Returns true if this expression is one that is sensitive to imperative assignment
    pub fn is_block_like(&self) -> bool {
        match self {
            Self::Match(Match { ref expr, .. }) => expr.is_block_like(),
            Self::Begin(_) | Self::If(_) | Self::Case(_) => true,
            _ => false,
        }
    }

    pub fn is_literal(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            Self::Cons(ref cons) => cons.is_literal(),
            Self::Tuple(ref tuple) => tuple.is_literal(),
            _ => false,
        }
    }

    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Self::Literal(ref lit) => Some(lit),
            _ => None,
        }
    }

    /// If this expression is an atom, this function returns the Ident
    /// backing the atom value. This is a common request in the compiler,
    /// hence its presence here
    pub fn as_atom(&self) -> Option<Ident> {
        match self {
            Self::Literal(Literal::Atom(a)) => Some(*a),
            _ => None,
        }
    }

    /// Same as `as_atom`, but unwraps the inner Symbol
    #[inline]
    pub fn as_atom_symbol(&self) -> Option<Symbol> {
        self.as_atom().map(|id| id.name)
    }

    /// Returns `Some(bool)` if the expression represents a literal boolean, otherwise None
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Literal(lit) => lit.as_boolean(),
            _ => None,
        }
    }

    pub fn as_var(&self) -> Option<Var> {
        match self {
            Self::Var(v) => Some(*v),
            _ => None,
        }
    }

    pub fn is_lc(&self) -> bool {
        match self {
            Self::ListComprehension(_) => true,
            _ => false,
        }
    }

    pub fn to_lc(self) -> ListComprehension {
        match self {
            Self::ListComprehension(lc) => lc,
            _ => panic!("not a list comprehension"),
        }
    }

    pub fn is_data_constructor(&self) -> bool {
        match self {
            Self::Literal(_) | Self::Cons(_) | Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub fn coerce_to_float(self) -> Expr {
        match self {
            expr @ Expr::Literal(Literal::Float(_, _)) => expr,
            Expr::Literal(Literal::Integer(span, i)) => {
                Expr::Literal(Literal::Float(span, Float::new(i.to_float()).unwrap()))
            }
            Expr::Literal(Literal::Char(span, c)) => {
                Expr::Literal(Literal::Float(span, Float::new(c as i64 as f64).unwrap()))
            }
            expr => expr,
        }
    }
}
impl From<Name> for Expr {
    fn from(name: Name) -> Self {
        match name {
            Name::Atom(ident) => Self::Literal(Literal::Atom(ident)),
            Name::Var(ident) => Self::Var(Var(ident)),
        }
    }
}
impl From<Span<Arity>> for Expr {
    fn from(arity: Span<Arity>) -> Self {
        match arity.item {
            Arity::Int(i) => Self::Literal(Literal::Integer(arity.span(), i.into())),
            Arity::Var(ident) => Self::Var(Var(ident)),
        }
    }
}
impl From<Span<FunctionName>> for Expr {
    fn from(name: Span<FunctionName>) -> Self {
        Self::FunctionVar(name.into())
    }
}
impl TryInto<Literal> for Expr {
    type Error = Expr;

    fn try_into(self) -> Result<Literal, Self::Error> {
        match self {
            Self::Literal(lit) => Ok(lit),
            Self::Tuple(tuple) => tuple.try_into().map_err(Expr::Tuple),
            Self::Cons(cons) => cons.try_into().map_err(Expr::Cons),
            Self::Map(map) => map.try_into().map_err(Expr::Map),
            other => Err(other),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Spanned)]
pub struct Var(pub Ident);
impl Var {
    #[inline]
    pub fn sym(&self) -> Symbol {
        self.0.name
    }

    #[inline]
    pub fn is_wildcard(&self) -> bool {
        self.0.name == symbols::Underscore
    }

    #[inline]
    pub fn is_wanted(&self) -> bool {
        self.0.as_str().get().starts_with('_') == false
    }

    #[inline]
    pub fn is_compiler_generated(&self) -> bool {
        self.0.as_str().get().starts_with('$')
    }
}
impl From<Ident> for Var {
    fn from(i: Ident) -> Self {
        Self(i)
    }
}
impl PartialOrd for Var {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Var {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.as_str().get().cmp(other.0.as_str().get())
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Cons {
    #[span]
    pub span: SourceSpan,
    pub head: Box<Expr>,
    pub tail: Box<Expr>,
}
impl Cons {
    pub fn is_literal(&self) -> bool {
        let mut current = Some(self);
        while let Some(Self { head, tail, .. }) = current.take() {
            if !head.is_literal() {
                return false;
            }
            match tail.as_ref() {
                Expr::Literal(_) => (),
                Expr::Tuple(ref tuple) if tuple.is_literal() => (),
                Expr::Map(ref map) if map.is_literal() => (),
                Expr::Cons(ref cons) => current = Some(cons),
                _ => return false,
            }
        }
        true
    }
}
impl TryInto<Literal> for Cons {
    type Error = Cons;

    fn try_into(self) -> Result<Literal, Self::Error> {
        let Cons { span, head, tail } = self;
        match (*head).try_into() {
            Ok(hd) => match (*tail).try_into() {
                Ok(tl) => Ok(Literal::Cons(span, Box::new(hd), Box::new(tl))),
                Err(tl) => Err(Cons {
                    span,
                    head: Box::new(Expr::Literal(hd)),
                    tail: Box::new(tl),
                }),
            },
            Err(hd) => Err(Cons {
                span,
                head: Box::new(hd),
                tail,
            }),
        }
    }
}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Tuple {
    #[span]
    pub span: SourceSpan,
    pub elements: Vec<Expr>,
}
impl Tuple {
    pub fn is_literal(&self) -> bool {
        self.elements.iter().all(|expr| expr.is_literal())
    }
}
impl TryInto<Literal> for Tuple {
    type Error = Tuple;

    fn try_into(mut self) -> Result<Literal, Self::Error> {
        if self.is_literal() {
            let elements = self
                .elements
                .drain(..)
                .map(|e| e.try_into().unwrap())
                .collect();
            Ok(Literal::Tuple(self.span, elements))
        } else {
            Err(self)
        }
    }
}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Map {
    #[span]
    pub span: SourceSpan,
    pub fields: Vec<MapField>,
}
impl Map {
    pub fn is_literal(&self) -> bool {
        for field in self.fields.iter() {
            if !field.key_ref().is_literal() {
                return false;
            }
            if !field.value_ref().is_literal() {
                return false;
            }
        }
        true
    }
}
impl TryInto<Literal> for Map {
    type Error = Map;

    fn try_into(mut self) -> Result<Literal, Self::Error> {
        if self.is_literal() {
            let mut map: BTreeMap<Literal, Literal> = BTreeMap::new();
            for field in self.fields.drain(..) {
                let key = field.key().try_into().unwrap();
                let value = field.value().try_into().unwrap();
                map.insert(key, value);
            }
            Ok(Literal::Map(self.span, map))
        } else {
            Err(self)
        }
    }
}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

// Updating fields on an existing map, e.g. `Map#{field1 = value1}.`
#[derive(Debug, Clone, Spanned)]
pub struct MapUpdate {
    #[span]
    pub span: SourceSpan,
    pub map: Box<Expr>,
    pub updates: Vec<MapField>,
}
impl PartialEq for MapUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map && self.updates == other.updates
    }
}

/// Maps can have two different types of field assignment:
///
/// * assoc - inserts or updates the given key with the given value
/// * exact - updates the given key with the given value, or produces an error
#[derive(Debug, Clone, Spanned)]
pub enum MapField {
    Assoc {
        #[span]
        span: SourceSpan,
        key: Expr,
        value: Expr,
    },
    Exact {
        #[span]
        span: SourceSpan,
        key: Expr,
        value: Expr,
    },
}
impl MapField {
    pub fn key(&self) -> Expr {
        self.key_ref().clone()
    }

    pub fn key_ref(&self) -> &Expr {
        match self {
            Self::Assoc { ref key, .. } => key,
            Self::Exact { ref key, .. } => key,
        }
    }

    pub fn value(&self) -> Expr {
        self.value_ref().clone()
    }

    pub fn value_ref(&self) -> &Expr {
        match self {
            Self::Assoc { ref value, .. } => value,
            Self::Exact { ref value, .. } => value,
        }
    }
}
impl PartialEq for MapField {
    fn eq(&self, other: &Self) -> bool {
        (self.key() == other.key()) && (self.value() == other.value())
    }
}

/// The set of literal values
#[derive(Debug, Clone, Spanned)]
pub enum Literal {
    Atom(Ident),
    String(Ident),
    Char(#[span] SourceSpan, char),
    Integer(#[span] SourceSpan, Int),
    Float(#[span] SourceSpan, Float),
    Nil(#[span] SourceSpan),
    Cons(#[span] SourceSpan, Box<Literal>, Box<Literal>),
    Tuple(#[span] SourceSpan, Vec<Literal>),
    Map(#[span] SourceSpan, BTreeMap<Literal, Literal>),
    Binary(#[span] SourceSpan, BitVec),
}
impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;
        match self {
            Self::Atom(id) => write!(f, "'{}'", id.name),
            Self::String(id) => write!(f, "\"{}\"", id.as_str().get().escape_debug()),
            Self::Char(_, c) => write!(f, "${}", c.escape_debug()),
            Self::Integer(_, i) => write!(f, "{}", i),
            Self::Float(_, flt) => write!(f, "{}", flt),
            Self::Nil(_) => write!(f, "[]"),
            Self::Cons(_, h, t) => {
                if let Ok(elements) = self.as_proper_list() {
                    f.write_char('[')?;
                    for (i, elem) in elements.iter().enumerate() {
                        if i > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    f.write_char(']')
                } else {
                    write!(f, "[{} | {}]", h, t)
                }
            }
            Self::Tuple(_, elements) => {
                f.write_char('{')?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                f.write_char('}')
            }
            Self::Map(_, map) => {
                f.write_str("#{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{} => {}", k, v)?;
                }
                f.write_char('}')
            }
            Self::Binary(_, bin) => write!(f, "{}", bin.display()),
        }
    }
}
impl Literal {
    pub fn from_proper_list(span: SourceSpan, mut elements: Vec<Literal>) -> Self {
        elements.drain(..).rfold(Self::Nil(span), |lit, tail| {
            Self::Cons(lit.span(), Box::new(lit), Box::new(tail))
        })
    }

    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Atom(id) => match id.name {
                symbols::True => Some(true),
                symbols::False => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<Int> {
        match self {
            Self::Integer(_, i) => Some(i.clone()),
            Self::Char(_, c) => Some(Int::Small(*c as i64)),
            _ => None,
        }
    }

    /// Converts this literal into a vector of elements representing a proper list
    pub fn as_proper_list(&self) -> Result<Vec<Literal>, ()> {
        match self {
            Self::String(s) => {
                let span = s.span;
                Ok(s.as_str()
                    .get()
                    .chars()
                    .map(|c| Literal::Integer(span, Int::Small(c as i64)))
                    .collect())
            }
            Self::Cons(_, head, tail) => {
                let mut elements = vec![];
                elements.push(head.as_ref().clone());
                let mut current = Some(tail.as_ref());
                while let Some(next) = current.take() {
                    match next {
                        // [H | "string"] =:= [H, $s, $t, $r, $i, $n, $g]
                        Self::String(s) => {
                            let span = s.span;
                            for c in s.as_str().get().chars() {
                                elements.push(Literal::Integer(span, Int::Small(c as i64)));
                            }
                        }
                        // End of list
                        Self::Nil(_) => {
                            break;
                        }
                        // [H | T]
                        Self::Cons(_, head, tail) => {
                            elements.push(head.as_ref().clone());
                            current = Some(tail.as_ref());
                        }
                        // Not a proper list
                        _ => return Err(()),
                    }
                }
                Ok(elements)
            }
            Self::Nil(_) => Ok(vec![]),
            // Not a list
            _ => Err(()),
        }
    }

    pub fn try_from_tuple(tuple: &Tuple) -> Option<Self> {
        if !tuple.is_literal() {
            return None;
        }

        let elements = tuple
            .elements
            .iter()
            .map(|expr| expr.as_literal().unwrap().clone())
            .collect();
        Some(Self::Tuple(tuple.span, elements))
    }
}
impl From<Number> for Literal {
    fn from(n: Number) -> Self {
        let span = SourceSpan::default();
        match n {
            Number::Integer(i) => Self::Integer(span, i),
            Number::Float(f) => Self::Float(span, f),
        }
    }
}
impl TryInto<Number> for Literal {
    type Error = Literal;

    fn try_into(self) -> Result<Number, Self::Error> {
        match self {
            Self::Integer(_, i) => Ok(i.into()),
            Self::Float(_, f) => Ok(f.into()),
            Self::Char(_, c) => Ok(Number::Integer(Int::Small(c as i64))),
            other => Err(other),
        }
    }
}
impl From<bool> for Literal {
    fn from(b: bool) -> Self {
        let span = SourceSpan::default();
        if b {
            Self::Atom(Ident::new(symbols::True, span))
        } else {
            Self::Atom(Ident::new(symbols::False, span))
        }
    }
}
impl Into<syntax_base::Literal> for Literal {
    fn into(self) -> syntax_base::Literal {
        match self {
            Self::Atom(id) => syntax_base::Literal::atom(id.span, id.name),
            Self::Char(span, c) => syntax_base::Literal::integer(span, c as i64),
            Self::Integer(span, i) => syntax_base::Literal::integer(span, i),
            Self::Float(span, f) => syntax_base::Literal::float(span, f),
            Self::Nil(span) => syntax_base::Literal::nil(span),
            Self::String(id) => {
                let span = id.span;
                id.as_str()
                    .get()
                    .chars()
                    .map(|c| syntax_base::Literal::integer(span, c as i64))
                    .rfold(syntax_base::Literal::nil(span), |tl, c| {
                        syntax_base::Literal::cons(span, c, tl)
                    })
            }
            Self::Cons(span, head, tail) => {
                syntax_base::Literal::cons(span, (*head).into(), (*tail).into())
            }
            Self::Tuple(span, mut elements) => {
                syntax_base::Literal::tuple(span, elements.drain(..).map(Self::into).collect())
            }
            Self::Map(span, mut map) => {
                let mut new_map: BTreeMap<syntax_base::Literal, syntax_base::Literal> =
                    BTreeMap::new();
                while let Some((k, v)) = map.pop_first() {
                    new_map.insert(k.into(), v.into());
                }
                syntax_base::Literal {
                    span,
                    annotations: Annotations::default(),
                    value: syntax_base::Lit::Map(new_map),
                }
            }
            Self::Binary(span, bin) => syntax_base::Literal {
                span,
                annotations: Annotations::default(),
                value: syntax_base::Lit::Binary(bin),
            },
        }
    }
}
impl PartialEq for Literal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Atom(x), Self::Atom(y)) => x.name == y.name,
            (Self::Atom(_), _) => false,
            (Self::String(x), Self::String(y)) => x.name == y.name,
            (x @ Self::String(_), y @ Self::Cons(_, _, _)) => {
                let xs = x.as_proper_list().unwrap();
                match y.as_proper_list() {
                    Ok(ys) => xs == ys,
                    Err(_) => false,
                }
            }
            (Self::String(_), _) => false,
            (Self::Char(_, x), Self::Char(_, y)) => x == y,
            (Self::Char(_, x), Self::Integer(_, y)) => {
                let x = Int::Small(*x as u32 as i64);
                x.eq(y)
            }
            (Self::Char(_, _), _) => false,
            (Self::Integer(_, x), Self::Integer(_, y)) => x == y,
            (Self::Integer(_, x), Self::Char(_, y)) => {
                let y = Int::Small(*y as u32 as i64);
                x.eq(&y)
            }
            (Self::Integer(_, _), _) => false,
            (Self::Float(_, x), Self::Float(_, y)) => x == y,
            (Self::Float(_, _), _) => false,
            (Self::Nil(_), Self::Nil(_)) => true,
            (Self::Nil(_), Self::String(s)) if s.name == symbols::Empty => true,
            (Self::Nil(_), _) => false,
            (Self::Cons(_, h1, t1), Self::Cons(_, h2, t2)) => h1 == h2 && t1 == t2,
            (x @ Self::Cons(_, _, _), y @ Self::String(_)) => {
                let ys = y.as_proper_list().unwrap();
                match x.as_proper_list() {
                    Ok(xs) => xs == ys,
                    Err(_) => false,
                }
            }
            (Self::Cons(_, _, _), _) => false,
            (Self::Tuple(_, xs), Self::Tuple(_, ys)) => xs == ys,
            (Self::Tuple(_, _), _) => false,
            (Self::Map(_, x), Self::Map(_, y)) => x == y,
            (Self::Map(_, _), _) => false,
            (Self::Binary(_, x), Self::Binary(_, y)) => x.eq(y),
            (Self::Binary(_, _), _) => false,
        }
    }
}
impl Eq for Literal {}
impl PartialOrd for Literal {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Float(_, x), Self::Float(_, y)) => x.partial_cmp(y),
            (Self::Float(_, x), Self::Integer(_, y)) => x.partial_cmp(y),
            (Self::Float(_, x), Self::Char(_, y)) => x.partial_cmp(&Int::Small(*y as u32 as i64)),
            (Self::Float(_, _), _) => Some(Ordering::Less),
            (Self::Integer(_, x), Self::Integer(_, y)) => x.partial_cmp(y),
            (Self::Integer(_, x), Self::Float(_, y)) => x.partial_cmp(y),
            (Self::Integer(_, x), Self::Char(_, y)) => x.partial_cmp(&Int::Small(*y as u32 as i64)),
            (Self::Integer(_, _), _) => Some(Ordering::Less),
            (Self::Char(_, x), Self::Integer(_, y)) => y.partial_cmp(x).map(|o| o.reverse()),
            (Self::Char(_, x), Self::Float(_, y)) => {
                let x = *x as u32 as i64;
                y.partial_cmp(&x).map(|o| o.reverse())
            }
            (Self::Char(_, x), Self::Char(_, y)) => y
                .partial_cmp(&Int::Small(*x as u32 as i64))
                .map(|o| o.reverse()),
            (Self::Char(_, _), _) => Some(Ordering::Less),
            (Self::Atom(_), Self::Float(_, _))
            | (Self::Atom(_), Self::Integer(_, _))
            | (Self::Atom(_), Self::Char(_, _)) => Some(Ordering::Greater),
            (Self::Atom(x), Self::Atom(y)) => x.partial_cmp(y),
            (Self::Atom(_), _) => Some(Ordering::Less),
            (Self::Tuple(_, _), Self::Float(_, _))
            | (Self::Tuple(_, _), Self::Integer(_, _))
            | (Self::Tuple(_, _), Self::Char(_, _))
            | (Self::Tuple(_, _), Self::Atom(_)) => Some(Ordering::Greater),
            (Self::Tuple(_, xs), Self::Tuple(_, ys)) => xs.partial_cmp(ys),
            (Self::Tuple(_, _), _) => Some(Ordering::Less),
            (Self::Map(_, _), Self::Float(_, _))
            | (Self::Map(_, _), Self::Integer(_, _))
            | (Self::Map(_, _), Self::Char(_, _))
            | (Self::Map(_, _), Self::Atom(_))
            | (Self::Map(_, _), Self::Tuple(_, _)) => Some(Ordering::Greater),
            (Self::Map(_, x), Self::Map(_, y)) => x.partial_cmp(y),
            (Self::Map(_, _), _) => Some(Ordering::Less),
            (Self::Nil(_), Self::Nil(_)) => Some(Ordering::Equal),
            (Self::Nil(_), Self::String(_)) | (Self::Nil(_), Self::Cons(_, _, _)) => {
                Some(Ordering::Less)
            }
            (Self::Nil(_), _) => Some(Ordering::Greater),
            (Self::String(s), Self::Nil(_)) if s.name == symbols::Empty => Some(Ordering::Equal),
            (Self::String(_), Self::Nil(_)) => Some(Ordering::Greater),
            (Self::String(x), Self::String(y)) => x.partial_cmp(y),
            (x @ Self::String(_), y @ Self::Cons(_, _, _)) => match y.as_proper_list() {
                Ok(ys) => {
                    let xs = x.as_proper_list().unwrap();
                    xs.partial_cmp(&ys)
                }
                Err(_) => Some(Ordering::Less),
            },
            (Self::String(_), _) => Some(Ordering::Greater),
            (Self::Cons(_, h1, t1), Self::Cons(_, h2, t2)) => match h1.partial_cmp(h2) {
                Some(Ordering::Equal) => t1.partial_cmp(t2),
                other => other,
            },
            (x @ Self::Cons(_, _, _), y @ Self::String(_)) => match x.as_proper_list() {
                Ok(xs) => {
                    let ys = y.as_proper_list().unwrap();
                    xs.partial_cmp(&ys)
                }
                Err(_) => Some(Ordering::Greater),
            },
            (Self::Cons(_, _, _), Self::Binary(_, _)) => Some(Ordering::Less),
            (Self::Cons(_, _, _), _) => Some(Ordering::Greater),
            (Self::Binary(_, x), Self::Binary(_, y)) => x.partial_cmp(y),
            (Self::Binary(_, _), _) => Some(Ordering::Greater),
        }
    }
}
impl Hash for Literal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Atom(x) => x.name.hash(state),
            Self::String(s) => s.name.hash(state),
            Self::Float(_, f) => f.hash(state),
            Self::Integer(_, i) => i.hash(state),
            Self::Char(_, c) => c.hash(state),
            Self::Nil(_) => (),
            Self::Cons(_, h, t) => {
                h.hash(state);
                t.hash(state);
            }
            Self::Tuple(_, elements) => Hash::hash_slice(elements.as_slice(), state),
            Self::Map(_, map) => map.hash(state),
            Self::Binary(_, bin) => bin.hash(state),
        }
    }
}
impl Ord for Literal {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Record {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub fields: Vec<RecordField>,
    pub default: Option<Box<Expr>>,
}
impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.fields == other.fields && self.default == other.default
    }
}

// Accessing a record field value, e.g. Expr#myrec.field1
#[derive(Debug, Clone, Spanned)]
pub struct RecordAccess {
    #[span]
    pub span: SourceSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordAccess {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.field == other.field && self.record == other.record
    }
}

// Referencing a record fields index, e.g. #myrec.field1
#[derive(Debug, Clone, Spanned)]
pub struct RecordIndex {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordIndex {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.field == other.field
    }
}

// Update a record field value, e.g. Expr#myrec{field1=ValueExpr}
#[derive(Debug, Clone, Spanned)]
pub struct RecordUpdate {
    #[span]
    pub span: SourceSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub updates: Vec<RecordField>,
}
impl PartialEq for RecordUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.record == other.record && self.updates == other.updates
    }
}

/// Record fields always have a name, but both default value and type
/// are optional in a record definition. When instantiating a record,
/// if no value is given for a field, and no default is given,
/// then `undefined` is the default.
#[derive(Debug, Clone, Spanned)]
pub struct RecordField {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub value: Option<Expr>,
    pub ty: Option<Type>,
    pub is_default: bool,
}
impl PartialEq for RecordField {
    fn eq(&self, other: &Self) -> bool {
        (self.name == other.name) && (self.value == other.value) && (self.ty == other.ty)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Binary {
    #[span]
    pub span: SourceSpan,
    pub elements: Vec<BinaryElement>,
}
impl PartialEq for Binary {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

/// Used to represent a specific segment in a binary constructor, to
/// produce a binary, all segments must be evaluated, and then assembled
#[derive(Debug, Clone, Spanned)]
pub struct BinaryElement {
    #[span]
    pub span: SourceSpan,
    pub bit_expr: Expr,
    pub bit_size: Option<Expr>,
    pub specifier: Option<BinaryEntrySpecifier>,
}
impl PartialEq for BinaryElement {
    fn eq(&self, other: &Self) -> bool {
        (self.bit_expr == other.bit_expr)
            && (self.bit_size == other.bit_size)
            && (self.specifier == other.specifier)
    }
}

/// A bit type can come in the form `Type` or `Type:Size`
#[derive(Debug, Clone, Spanned)]
pub enum BitType {
    Name(#[span] SourceSpan, Ident),
    Sized(#[span] SourceSpan, Ident, usize),
}
impl PartialEq for BitType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&BitType::Name(_, ref x1), &BitType::Name(_, ref y1)) => x1 == y1,
            (&BitType::Sized(_, ref x1, ref x2), &BitType::Sized(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ListComprehension {
    #[span]
    pub span: SourceSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for ListComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.qualifiers == other.qualifiers
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct BinaryComprehension {
    #[span]
    pub span: SourceSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for BinaryComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.qualifiers == other.qualifiers
    }
}

/// A generator is one of two types of expressions that act as qualifiers in a commprehension, the other is a filter
#[derive(Debug, Clone, Spanned)]
pub struct Generator {
    #[span]
    pub span: SourceSpan,
    pub ty: GeneratorType,
    pub pattern: Box<Expr>,
    pub expr: Box<Expr>,
}
impl PartialEq for Generator {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.expr == other.expr
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GeneratorType {
    Default,
    Bitstring,
}
impl Default for GeneratorType {
    fn default() -> Self {
        Self::Default
    }
}

// A sequence of expressions, e.g. begin expr1, .., exprN end
#[derive(Debug, Clone, Spanned)]
pub struct Begin {
    #[span]
    pub span: SourceSpan,
    pub body: Vec<Expr>,
}
impl PartialEq for Begin {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

// Function application, e.g. foo(expr1, .., exprN)
#[derive(Debug, Clone, Spanned)]
pub struct Apply {
    #[span]
    pub span: SourceSpan,
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}
impl Apply {
    pub fn new(span: SourceSpan, callee: Expr, args: Vec<Expr>) -> Self {
        Self {
            span,
            callee: Box::new(callee),
            args,
        }
    }

    pub fn remote(span: SourceSpan, module: Symbol, function: Symbol, args: Vec<Expr>) -> Self {
        Self {
            span,
            callee: Box::new(Expr::FunctionVar(FunctionVar::new(
                span,
                module,
                function,
                args.len().try_into().unwrap(),
            ))),
            args,
        }
    }

    pub fn local(span: SourceSpan, function: Symbol, args: Vec<Expr>) -> Self {
        Self {
            span,
            callee: Box::new(Expr::FunctionVar(FunctionVar::new_local(
                span,
                function,
                args.len().try_into().unwrap(),
            ))),
            args,
        }
    }
}
impl PartialEq for Apply {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args
    }
}

// Remote, e.g. Foo:Bar
#[derive(Debug, Clone, Spanned)]
pub struct Remote {
    #[span]
    pub span: SourceSpan,
    pub module: Box<Expr>,
    pub function: Box<Expr>,
}
impl Remote {
    pub fn new(span: SourceSpan, module: Expr, function: Expr) -> Self {
        Self {
            span,
            module: Box::new(module),
            function: Box::new(function),
        }
    }

    pub fn new_literal(span: SourceSpan, module: Symbol, function: Symbol) -> Self {
        Self {
            span,
            module: Box::new(Expr::Literal(Literal::Atom(Ident::new(module, span)))),
            function: Box::new(Expr::Literal(Literal::Atom(Ident::new(function, span)))),
        }
    }

    /// Try to resolve this remote expression to a constant function reference of the given arity
    pub fn try_eval(&self, arity: u8) -> Result<FunctionName, EvalError> {
        let span = self.span;
        let module = evaluator::eval_expr(self.module.as_ref(), None)?;
        let function = evaluator::eval_expr(self.function.as_ref(), None)?;
        match (module, function) {
            (Literal::Atom(m), Literal::Atom(f)) => Ok(FunctionName::new(m.name, f.name, arity)),
            _ => Err(EvalError::InvalidConstExpression { span }),
        }
    }
}
impl PartialEq for Remote {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct BinaryExpr {
    #[span]
    pub span: SourceSpan,
    pub lhs: Box<Expr>,
    pub op: BinaryOp,
    pub rhs: Box<Expr>,
}
impl BinaryExpr {
    pub fn new(span: SourceSpan, op: BinaryOp, lhs: Expr, rhs: Expr) -> Self {
        Self {
            span,
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}
impl PartialEq for BinaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.lhs == other.lhs && self.rhs == other.rhs
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct UnaryExpr {
    #[span]
    pub span: SourceSpan,
    pub op: UnaryOp,
    pub operand: Box<Expr>,
}
impl PartialEq for UnaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.operand == other.operand
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Match {
    #[span]
    pub span: SourceSpan,
    pub pattern: Box<Expr>,
    pub expr: Box<Expr>,
}
impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.expr == other.expr
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct If {
    #[span]
    pub span: SourceSpan,
    pub clauses: Vec<Clause>,
}
impl If {
    /// Returns true if the last clause of the `if` is the literal boolean `true`
    pub fn has_wildcard_clause(&self) -> bool {
        self.clauses
            .last()
            .map(|clause| clause.is_wildcard())
            .unwrap_or(false)
    }
}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Catch {
    #[span]
    pub span: SourceSpan,
    pub expr: Box<Expr>,
}
impl PartialEq for Catch {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Case {
    #[span]
    pub span: SourceSpan,
    pub expr: Box<Expr>,
    pub clauses: Vec<Clause>,
}
impl PartialEq for Case {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr && self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Receive {
    #[span]
    pub span: SourceSpan,
    pub clauses: Option<Vec<Clause>>,
    pub after: Option<After>,
}
impl PartialEq for Receive {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses && self.after == other.after
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Try {
    #[span]
    pub span: SourceSpan,
    pub exprs: Vec<Expr>,
    pub clauses: Option<Vec<Clause>>,
    pub catch_clauses: Option<Vec<Clause>>,
    pub after: Option<Vec<Expr>>,
}
impl PartialEq for Try {
    fn eq(&self, other: &Self) -> bool {
        self.exprs == other.exprs
            && self.clauses == other.clauses
            && self.catch_clauses == other.catch_clauses
            && self.after == other.after
    }
}

/// Represents the `after` clause of a `receive` expression
#[derive(Debug, Clone, Spanned)]
pub struct After {
    #[span]
    pub span: SourceSpan,
    pub timeout: Box<Expr>,
    pub body: Vec<Expr>,
}
impl PartialEq for After {
    fn eq(&self, other: &Self) -> bool {
        self.timeout == other.timeout && self.body == other.body
    }
}

/// Represents a single match clause in a `case`, `try`, or `receive` expression
#[derive(Debug, Clone, Spanned)]
pub struct Clause {
    #[span]
    pub span: SourceSpan,
    pub patterns: Vec<Expr>,
    pub guards: Vec<Guard>,
    pub body: Vec<Expr>,
    pub compiler_generated: bool,
}
impl Clause {
    pub fn new(
        span: SourceSpan,
        patterns: Vec<Expr>,
        guards: Vec<Guard>,
        body: Vec<Expr>,
        compiler_generated: bool,
    ) -> Self {
        Self {
            span,
            patterns,
            guards,
            body,
            compiler_generated,
        }
    }

    pub fn for_if(
        span: SourceSpan,
        guards: Vec<Guard>,
        body: Vec<Expr>,
        compiler_generated: bool,
    ) -> Self {
        Self {
            span,
            patterns: vec![Expr::Var(Var(Ident::new(symbols::Underscore, span)))],
            guards,
            body,
            compiler_generated,
        }
    }

    pub fn for_catch(
        span: SourceSpan,
        kind: Expr,
        error: Expr,
        trace: Option<Expr>,
        guards: Vec<Guard>,
        body: Vec<Expr>,
    ) -> Self {
        let trace = trace.unwrap_or_else(|| Expr::Var(Var(Ident::from_str("_"))));
        Self {
            span,
            patterns: vec![kind, error, trace],
            guards,
            body,
            compiler_generated: false,
        }
    }

    pub fn is_wildcard(&self) -> bool {
        let is_wild = self.patterns.iter().all(|p| {
            if let Expr::Var(v) = p {
                v.is_wildcard()
            } else {
                false
            }
        });
        if is_wild {
            match self.guards.len() {
                0 => true,
                1 => self
                    .guards
                    .first()
                    .and_then(|g| g.as_boolean())
                    .unwrap_or_default(),
                _ => false,
            }
        } else {
            false
        }
    }
}
impl PartialEq for Clause {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns && self.guards == other.guards && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Protect {
    #[span]
    pub span: SourceSpan,
    pub body: Box<Expr>,
}
impl PartialEq for Protect {
    fn eq(&self, other: &Self) -> bool {
        self.body.eq(&other.body)
    }
}
