use rpds::RedBlackTreeSet;

use crate::Annotated;
use firefly_intern::Ident;

/// Return the set of variables that are used in at least one of the given annotated items
pub fn used_in_any<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(RedBlackTreeSet::new(), |used, annotated| {
        union(annotated.used_vars(), used)
    })
}

/// Return the set of variables that are new in at least one of the given annotated items
pub fn new_in_any<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(RedBlackTreeSet::new(), |new, annotated| {
        union(annotated.new_vars(), new)
    })
}

/// Return the set of variables that are new in every one of the given annotated items
pub fn new_in_all<'a, A: Annotated + 'a, I: Iterator<Item = &'a A>>(
    iter: I,
) -> RedBlackTreeSet<Ident> {
    iter.fold(None, |new, annotated| match new {
        None => Some(annotated.new_vars().clone()),
        Some(ns) => Some(intersection(annotated.new_vars(), ns)),
    })
    .unwrap_or_default()
}

/// Return the union of two sets of variables
pub fn union(x: RedBlackTreeSet<Ident>, y: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    let mut result = x;
    for id in y.iter().copied() {
        result.insert_mut(id);
    }
    result
}

/// Return the set of variables in `x` that are not in `y`
pub fn subtract(x: RedBlackTreeSet<Ident>, y: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    let mut result = x;
    for id in y.iter() {
        result.remove_mut(id);
    }
    result
}

/// Return the intersection of two sets of variables
pub fn intersection(
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
