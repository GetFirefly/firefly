use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{once, FromIterator, IntoIterator};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub struct Seq<T> {
    vec: Option<Arc<Vec<T>>>,
}
unsafe impl<T> Send for Seq<T> {}
impl<T> Seq<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extend this `Seq`. Note that seqs from which this was cloned
    /// are not affected.
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
        T: Clone,
    {
        let mut iter = iter.into_iter();

        // Peel off the first item; if there is none, then just do nothing.
        if let Some(first) = iter.next() {
            // Create an iterator with that first item plus the rest
            let iter = once(first).chain(iter);

            // Try to extend in place
            if let Some(vec) = &mut self.vec {
                Arc::make_mut(vec).extend(iter);
                return;
            }

            // If not, construct a new vector.
            let v: Vec<T> = self.iter().cloned().chain(iter).collect();
            self.vec = Some(Arc::new(v));
        }
    }
}

impl<T> Clone for Seq<T> {
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
        }
    }
}

impl<T> Default for Seq<T> {
    fn default() -> Self {
        Self { vec: None }
    }
}

impl<T> From<Arc<Vec<T>>> for Seq<T> {
    fn from(vec: Arc<Vec<T>>) -> Self {
        if vec.is_empty() {
            Seq::default()
        } else {
            Seq { vec: Some(vec) }
        }
    }
}

impl<T> From<Vec<T>> for Seq<T> {
    fn from(vec: Vec<T>) -> Self {
        if vec.is_empty() {
            Seq::default()
        } else {
            Self::from(Arc::new(vec))
        }
    }
}

impl<T: Clone> From<&[T]> for Seq<T> {
    fn from(text: &[T]) -> Self {
        let vec: Vec<T> = text.iter().cloned().collect();
        Self::from(vec)
    }
}

impl<T> Deref for Seq<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.vec {
            None => &[],
            Some(vec) => vec.as_slice(),
        }
    }
}

impl<T: Clone> DerefMut for Seq<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.vec {
            None => &mut [],
            Some(ref mut vec) => Arc::make_mut(vec).as_mut_slice(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Seq<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <[T] as fmt::Debug>::fmt(self, fmt)
    }
}

impl<T: PartialEq> PartialEq<Seq<T>> for Seq<T> {
    fn eq(&self, other: &Seq<T>) -> bool {
        let this: &[T] = self;
        let other: &[T] = other;
        this == other
    }
}

impl<T: Eq> Eq for Seq<T> {}

impl<T: PartialEq> PartialEq<[T]> for Seq<T> {
    fn eq(&self, other: &[T]) -> bool {
        let this: &[T] = self;
        this == other
    }
}

impl<T: PartialEq> PartialEq<Vec<T>> for Seq<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        let this: &[T] = self;
        let other: &[T] = other;
        this == other
    }
}

impl<A: ?Sized, T: PartialEq> PartialEq<&A> for Seq<T>
where
    Seq<T>: PartialEq<A>,
{
    fn eq(&self, other: &&A) -> bool {
        self == *other
    }
}

impl<T> FromIterator<T> for Seq<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vec: Vec<T> = iter.into_iter().collect();
        Seq::from(vec)
    }
}

impl<'seq, T> IntoIterator for &'seq Seq<T> {
    type IntoIter = <&'seq [T] as IntoIterator>::IntoIter;
    type Item = <&'seq [T] as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'seq, T: Clone> IntoIterator for &'seq mut Seq<T> {
    type IntoIter = <&'seq mut [T] as IntoIterator>::IntoIter;
    type Item = <&'seq mut [T] as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Hash> Hash for Seq<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        <[T] as Hash>::hash(self, state)
    }
}

#[macro_export]
macro_rules! seq {
    () => {
        Seq::default()
    };

    ($($v:expr),* $(,)*) => {
        Seq::from(vec![$($v),*])
    };
}
