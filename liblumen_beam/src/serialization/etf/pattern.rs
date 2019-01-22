use std::fmt::Debug;

use num::bigint::ToBigInt;
use num::bigint::ToBigUint;
use num::traits::ToPrimitive;

use self::convert::AsOption;
use self::convert::TryAsRef;
use super::*;

pub type Result<'a, T> = std::result::Result<T, Unmatch<'a>>;

pub trait Pattern<'a>: Debug + Clone {
    type Output;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output>;

    fn unmatched(&self, input: &'a Term) -> Unmatch<'a>
    where
        Self: 'static,
    {
        Unmatch {
            input,
            pattern: Box::new(self.clone()),
            cause: None,
        }
    }
}

#[derive(Debug)]
pub struct Unmatch<'a> {
    pub input: &'a Term,
    pub pattern: Box<Debug>,
    pub cause: Option<Box<Unmatch<'a>>>,
}
impl<'a> Unmatch<'a> {
    pub fn cause(mut self, cause: Unmatch<'a>) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
    pub fn depth(&self) -> usize {
        let mut depth = 0;
        let mut curr = &self.cause;
        while let Some(ref next) = *curr {
            depth += 1;
            curr = &next.cause;
        }
        depth
    }
    pub fn max_depth(self, other: Self) -> Self {
        if self.depth() < other.depth() {
            other
        } else {
            self
        }
    }
}

#[derive(Debug, Clone)]
pub enum Union2<A, B> {
    A(A),
    B(B),
}
impl<A, B> Union2<A, B> {
    pub fn is_a(&self) -> bool {
        match *self {
            Union2::A(_) => true,
            _ => false,
        }
    }
    pub fn is_b(&self) -> bool {
        match *self {
            Union2::B(_) => true,
            _ => false,
        }
    }
    pub fn into_result(self) -> ::std::result::Result<A, B> {
        match self {
            Union2::A(x) => Ok(x),
            Union2::B(x) => Err(x),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Union3<A, B, C> {
    A(A),
    B(B),
    C(C),
}

#[derive(Debug, Clone)]
pub enum Union4<A, B, C, D> {
    A(A),
    B(B),
    C(C),
    D(D),
}

#[derive(Debug, Clone)]
pub enum Union5<A, B, C, D, E> {
    A(A),
    B(B),
    C(C),
    D(D),
    E(E),
}

#[derive(Debug, Clone)]
pub enum Union6<A, B, C, D, E, F> {
    A(A),
    B(B),
    C(C),
    D(D),
    E(E),
    F(F),
}

#[derive(Debug, Clone)]
pub struct Any<T>(::std::marker::PhantomData<T>);
impl<T> Any<T>
where
    T: Debug,
{
    pub fn new() -> Self {
        Any(::std::marker::PhantomData)
    }
}
pub fn any<T>() -> Any<T>
where
    T: Debug,
{
    Any::new()
}
impl<'a, O> Pattern<'a> for Any<O>
where
    O: Debug + Clone + 'static,
    Term: TryAsRef<O>,
{
    type Output = &'a O;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.try_as_ref().ok_or_else(|| self.unmatched(input))
    }
}

impl<'a> Pattern<'a> for &'static str {
    type Output = Self;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let a: &Atom = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        (*self == a.name)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        Ok(*self)
    }
}

#[derive(Debug, Clone)]
pub struct VarList<P>(pub P);
impl<'a, P> Pattern<'a> for VarList<P>
where
    P: Pattern<'a> + 'static,
{
    type Output = Vec<P::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let mut outputs = Vec::with_capacity(l.elements.len());
        for e in &l.elements {
            outputs.push(
                self.0
                    .try_match(e)
                    .map_err(|e| self.unmatched(input).cause(e))?,
            );
        }
        Ok(outputs)
    }
}

#[derive(Debug, Clone)]
pub struct FixList<T>(pub T);
impl<'a, P0> Pattern<'a> for FixList<(P0,)>
where
    P0: Pattern<'a> + 'static,
{
    type Output = P0::Output;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 1)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok(o0)
    }
}

impl<'a, P0, P1> Pattern<'a> for FixList<(P0, P1)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 2)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = (self.0)
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1))
    }
}

impl<'a, P0, P1, P2> Pattern<'a> for FixList<(P0, P1, P2)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 3)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = (self.0)
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = (self.0)
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2))
    }
}

impl<'a, P0, P1, P2, P3> Pattern<'a> for FixList<(P0, P1, P2, P3)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output, P3::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 4)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = (self.0)
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = (self.0)
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = (self.0)
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3))
    }
}

impl<'a, P0, P1, P2, P3, P4> Pattern<'a> for FixList<(P0, P1, P2, P3, P4)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output, P3::Output, P4::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 5)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = (self.0)
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = (self.0)
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = (self.0)
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o4 = (self.0)
            .4
            .try_match(&e[4])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3, o4))
    }
}

impl<'a, P0, P1, P2, P3, P4, P5> Pattern<'a> for FixList<(P0, P1, P2, P3, P4, P5)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
    P5: Pattern<'a> + 'static,
{
    type Output = (
        P0::Output,
        P1::Output,
        P2::Output,
        P3::Output,
        P4::Output,
        P5::Output,
    );
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() == 6)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = (self.0)
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = (self.0)
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = (self.0)
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = (self.0)
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o4 = (self.0)
            .4
            .try_match(&e[4])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o5 = (self.0)
            .5
            .try_match(&e[5])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3, o4, o5))
    }
}

#[derive(Debug, Clone)]
pub struct Nil;
impl<'a> Pattern<'a> for Nil {
    type Output = &'a [Term];
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        (l.elements.len() == 0)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        Ok(&l.elements)
    }
}

#[derive(Debug, Clone)]
pub struct Cons<H, T>(pub H, pub T);
impl<'a, P0, P1> Pattern<'a> for Cons<P0, P1>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
{
    type Output = (P0::Output, Vec<P1::Output>);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &l.elements;
        (e.len() > 0)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let h = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;

        let mut tail = Vec::with_capacity(l.elements.len() - 1);
        for e in &l.elements[1..] {
            tail.push(
                self.1
                    .try_match(e)
                    .map_err(|e| self.unmatched(input).cause(e))?,
            );
        }
        Ok((h, tail))
    }
}

impl<'a> Pattern<'a> for () {
    type Output = ();
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        (t.elements.len() == 0)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        Ok(())
    }
}

impl<'a, P0> Pattern<'a> for (P0,)
where
    P0: Pattern<'a> + 'static,
{
    type Output = P0::Output;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        (t.elements.len() == 1)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&t.elements[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok(o0)
    }
}

impl<'a, P0, P1> Pattern<'a> for (P0, P1)
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &t.elements;
        (e.len() == 2)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = self
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1))
    }
}

impl<'a, P0, P1, P2> Pattern<'a> for (P0, P1, P2)
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &t.elements;
        (e.len() == 3)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = self
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = self
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2))
    }
}

impl<'a, P0, P1, P2, P3> Pattern<'a> for (P0, P1, P2, P3)
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output, P3::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &t.elements;
        (e.len() == 4)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = self
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = self
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = self
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3))
    }
}

impl<'a, P0, P1, P2, P3, P4> Pattern<'a> for (P0, P1, P2, P3, P4)
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
{
    type Output = (P0::Output, P1::Output, P2::Output, P3::Output, P4::Output);
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &t.elements;
        (e.len() == 5)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = self
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = self
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = self
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o4 = self
            .4
            .try_match(&e[4])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3, o4))
    }
}

impl<'a, P0, P1, P2, P3, P4, P5> Pattern<'a> for (P0, P1, P2, P3, P4, P5)
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
    P5: Pattern<'a> + 'static,
{
    type Output = (
        P0::Output,
        P1::Output,
        P2::Output,
        P3::Output,
        P4::Output,
        P5::Output,
    );
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let t: &Tuple = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let e = &t.elements;
        (e.len() == 6)
            .as_option()
            .ok_or_else(|| self.unmatched(input))?;
        let o0 = self
            .0
            .try_match(&e[0])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o1 = self
            .1
            .try_match(&e[1])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o2 = self
            .2
            .try_match(&e[2])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o3 = self
            .3
            .try_match(&e[3])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o4 = self
            .4
            .try_match(&e[4])
            .map_err(|e| self.unmatched(input).cause(e))?;
        let o5 = self
            .5
            .try_match(&e[5])
            .map_err(|e| self.unmatched(input).cause(e))?;
        Ok((o0, o1, o2, o3, o4, o5))
    }
}

macro_rules! try_err {
    ($e:expr) => {
        match $e {
            Ok(value) => return Ok(value),
            Err(err) => err,
        }
    };
}

#[derive(Debug, Clone)]
pub struct Or<T>(pub T);
impl<'a, P0, P1> Pattern<'a> for Or<(P0, P1)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
{
    type Output = Union2<P0::Output, P1::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let e = try_err!((self.0).0.try_match(input).map(|o| Union2::A(o)));
        let e = try_err!((self.0).1.try_match(input).map(|o| Union2::B(o))).max_depth(e);
        Err(self.unmatched(input).cause(e))
    }
}
impl<'a, P0, P1, P2> Pattern<'a> for Or<(P0, P1, P2)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
{
    type Output = Union3<P0::Output, P1::Output, P2::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let e = try_err!((self.0).0.try_match(input).map(|o| Union3::A(o)));
        let e = try_err!((self.0).1.try_match(input).map(|o| Union3::B(o))).max_depth(e);
        let e = try_err!((self.0).2.try_match(input).map(|o| Union3::C(o))).max_depth(e);
        Err(self.unmatched(input).cause(e))
    }
}
impl<'a, P0, P1, P2, P3> Pattern<'a> for Or<(P0, P1, P2, P3)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
{
    type Output = Union4<P0::Output, P1::Output, P2::Output, P3::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let e = try_err!((self.0).0.try_match(input).map(|o| Union4::A(o)));
        let e = try_err!((self.0).1.try_match(input).map(|o| Union4::B(o))).max_depth(e);
        let e = try_err!((self.0).2.try_match(input).map(|o| Union4::C(o))).max_depth(e);
        let e = try_err!((self.0).3.try_match(input).map(|o| Union4::D(o))).max_depth(e);
        Err(self.unmatched(input).cause(e))
    }
}
impl<'a, P0, P1, P2, P3, P4> Pattern<'a> for Or<(P0, P1, P2, P3, P4)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
{
    type Output = Union5<P0::Output, P1::Output, P2::Output, P3::Output, P4::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let e = try_err!((self.0).0.try_match(input).map(|o| Union5::A(o)));
        let e = try_err!((self.0).1.try_match(input).map(|o| Union5::B(o))).max_depth(e);
        let e = try_err!((self.0).2.try_match(input).map(|o| Union5::C(o))).max_depth(e);
        let e = try_err!((self.0).3.try_match(input).map(|o| Union5::D(o))).max_depth(e);
        let e = try_err!((self.0).4.try_match(input).map(|o| Union5::E(o))).max_depth(e);
        Err(self.unmatched(input).cause(e))
    }
}
impl<'a, P0, P1, P2, P3, P4, P5> Pattern<'a> for Or<(P0, P1, P2, P3, P4, P5)>
where
    P0: Pattern<'a> + 'static,
    P1: Pattern<'a> + 'static,
    P2: Pattern<'a> + 'static,
    P3: Pattern<'a> + 'static,
    P4: Pattern<'a> + 'static,
    P5: Pattern<'a> + 'static,
{
    type Output = Union6<P0::Output, P1::Output, P2::Output, P3::Output, P4::Output, P5::Output>;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let e = try_err!((self.0).0.try_match(input).map(|o| Union6::A(o)));
        let e = try_err!((self.0).1.try_match(input).map(|o| Union6::B(o))).max_depth(e);
        let e = try_err!((self.0).2.try_match(input).map(|o| Union6::C(o))).max_depth(e);
        let e = try_err!((self.0).3.try_match(input).map(|o| Union6::D(o))).max_depth(e);
        let e = try_err!((self.0).4.try_match(input).map(|o| Union6::E(o))).max_depth(e);
        let e = try_err!((self.0).5.try_match(input).map(|o| Union6::F(o))).max_depth(e);
        Err(self.unmatched(input).cause(e))
    }
}

#[derive(Debug, Clone)]
pub struct Ascii;
impl<'a> Pattern<'a> for Ascii {
    type Output = char;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let n = input.to_u8().ok_or_else(|| self.unmatched(input))?;
        if n < 0x80 {
            Ok(n as char)
        } else {
            Err(self.unmatched(input))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Unicode;
impl<'a> Pattern<'a> for Unicode {
    type Output = char;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let n = input.to_u32().ok_or_else(|| self.unmatched(input))?;
        ::std::char::from_u32(n).ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct Str<C>(pub C);
impl<'a, C> Pattern<'a> for Str<C>
where
    C: Pattern<'a, Output = char> + 'static,
{
    type Output = String;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        let l: &List = input.try_as_ref().ok_or_else(|| self.unmatched(input))?;
        let mut s = String::with_capacity(l.elements.len());
        for e in &l.elements {
            let c = self
                .0
                .try_match(e)
                .map_err(|e| self.unmatched(input).cause(e))?;
            s.push(c);
        }
        Ok(s)
    }
}

#[derive(Debug, Clone)]
pub struct U8;
impl<'a> Pattern<'a> for U8 {
    type Output = u8;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_u8().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct I8;
impl<'a> Pattern<'a> for I8 {
    type Output = i8;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_i8().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct U16;
impl<'a> Pattern<'a> for U16 {
    type Output = u16;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_u16().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct I16;
impl<'a> Pattern<'a> for I16 {
    type Output = i16;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_i16().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct U32;
impl<'a> Pattern<'a> for U32 {
    type Output = u32;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_u32().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct I32;
impl<'a> Pattern<'a> for I32 {
    type Output = i32;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_i32().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct U64;
impl<'a> Pattern<'a> for U64 {
    type Output = u64;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_u64().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct I64;
impl<'a> Pattern<'a> for I64 {
    type Output = i64;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_i64().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct Int;
impl<'a> Pattern<'a> for Int {
    type Output = num::BigInt;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_bigint().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct Uint;
impl<'a> Pattern<'a> for Uint {
    type Output = num::BigUint;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_biguint().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct F32;
impl<'a> Pattern<'a> for F32 {
    type Output = f32;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_f32().ok_or_else(|| self.unmatched(input))
    }
}

#[derive(Debug, Clone)]
pub struct F64;
impl<'a> Pattern<'a> for F64 {
    type Output = f64;
    fn try_match(&self, input: &'a Term) -> Result<'a, Self::Output> {
        input.to_f64().ok_or_else(|| self.unmatched(input))
    }
}
