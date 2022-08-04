mod ast;
mod cst;
mod kst;

pub use self::ast::CanonicalizeSyntax;
pub use self::cst::AstToCst;
pub use self::kst::CstToKernel;

use crate::cst::Annotated;
use liblumen_intern::Ident;
use rpds::RedBlackTreeSet;

pub(self) fn used_in_any<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(RedBlackTreeSet::new(), |used, annotated| {
        union(annotated.used_vars(), used)
    })
}

pub(self) fn new_in_any<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(RedBlackTreeSet::new(), |new, annotated| {
        union(annotated.new_vars(), new)
    })
}

pub(self) fn new_in_all<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(None, |new, annotated| match new {
        None => Some(annotated.new_vars().clone()),
        Some(ns) => Some(intersection(annotated.new_vars(), ns)),
    })
    .unwrap_or_default()
}

pub(self) fn union(x: RedBlackTreeSet<Ident>, y: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    let mut result = x;
    for id in y.iter().copied() {
        result.insert_mut(id);
    }
    result
}

pub(self) fn subtract(
    x: RedBlackTreeSet<Ident>,
    y: RedBlackTreeSet<Ident>,
) -> RedBlackTreeSet<Ident> {
    let mut result = x;
    for id in y.iter() {
        result.remove_mut(id);
    }
    result
}

pub(self) fn intersection(
    x: RedBlackTreeSet<Ident>,
    y: RedBlackTreeSet<Ident>,
) -> RedBlackTreeSet<Ident> {
    let mut result = RedBlackTreeSet::new();
    for id in x.iter().copied() {
        if y.contains(&id) {
            result.insert_mut(id);
        }
    }
    result
}
