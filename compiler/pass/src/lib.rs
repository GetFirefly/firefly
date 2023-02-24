//! This crate provides the abstractions necessary to construct
//! data structure agnostic compiler passes. It enables you to:
//!
//! * Construct a pass that receives an arbitrary input, and produces arbitrary output
//! * Construct a pass pipeline that chains passes, taking as input the first passes input type,
//! and outputing the last passes' output type. With this, you can represent lowering through
//! various intermediate representations using a single pass pipeline.
//!
// This feature is only used for tests, and can be removed with minimal refactoring,
// but I'm in a rush and we're using nightly right now anyway
#![feature(box_patterns)]

/// This trait represents anything that can be run as a pass.
///
/// Passes operate on an input value, and return either the same type, or a new type, depending on the nature of the pass.
///
/// Implementations may represent a single pass, or an arbitrary number of passes that will be run as a single unit.
///
/// Functions are valid implementations of `Pass` as long as their signature is `fn<I, O>(I) -> Result<O, ()>`.
pub trait Pass {
    type Input<'a>;
    type Output<'a>;

    /// Runs the pass on the given input
    ///
    /// Errors should be reported via the registered error handler,
    /// Passes should return `Err` to signal that the pass has failed
    /// and compilation should be aborted
    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>>;

    /// Chains two passes together to form a new, fused pass
    fn chain<P>(self, pass: P) -> Chain<Self, P>
    where
        Self: Sized,
        P: for<'a> Pass<Input<'a> = Self::Output<'a>>,
    {
        Chain::new(self, pass)
    }
}
impl<P, T, U> Pass for &mut P
where
    P: for<'a> Pass<Input<'a> = T, Output<'a> = U>,
{
    type Input<'a> = T;
    type Output<'a> = U;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        (*self).run(input)
    }
}
impl<T, U, P> Pass for Box<P>
where
    P: ?Sized + for<'a> Pass<Input<'a> = T, Output<'a> = U>,
{
    type Input<'a> = T;
    type Output<'a> = U;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        (**self).run(input)
    }
}
impl<T, U> Pass for dyn FnMut(T) -> anyhow::Result<U> {
    type Input<'a> = T;
    type Output<'a> = U;

    #[inline]
    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        self(input)
    }
}

/// This struct is not meant to be used directly, but is instead produced
/// when chaining `Pass` implementations together. `Chain` itself implements `Pass`,
/// which is what enables us to chain together arbitrarily many passes into a single one.
pub struct Chain<A, B> {
    a: A,
    b: B,
}
impl<A, B> Chain<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}
impl<A, B> Clone for Chain<A, B>
where
    A: Clone,
    B: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.a.clone(), self.b.clone())
    }
}
impl<A, B> Pass for Chain<A, B>
where
    A: for<'a> Pass,
    B: for<'a> Pass<Input<'a> = <A as Pass>::Output<'a>>,
{
    type Input<'a> = <A as Pass>::Input<'a>;
    type Output<'a> = <B as Pass>::Output<'a>;

    fn run<'a>(&mut self, input: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let u = self.a.run(input)?;
        self.b.run(u)
    }
}
