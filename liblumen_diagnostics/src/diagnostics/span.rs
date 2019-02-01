use std::cmp::Ordering;
use std::{cmp, fmt};

use super::index::{ByteIndex, Index};

/// A region of code in a source file
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Span<I> {
    start: I,
    end: I,
}
impl<I: Ord> Span<I> {
    /// Create a new span
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let span = Span::new(ByteIndex(3), ByteIndex(6));
    /// assert_eq!(span.start(), ByteIndex(3));
    /// assert_eq!(span.end(), ByteIndex(6));
    /// ```
    ///
    /// `start` and `end` are reordered to maintain the invariant that `start <= end`
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let span = Span::new(ByteIndex(6), ByteIndex(3));
    /// assert_eq!(span.start(), ByteIndex(3));
    /// assert_eq!(span.end(), ByteIndex(6));
    /// ```
    #[inline]
    pub fn new(start: I, end: I) -> Span<I> {
        if start <= end {
            Span { start, end }
        } else {
            Span {
                start: end,
                end: start,
            }
        }
    }

    #[inline]
    pub fn map<F, J>(self, mut f: F) -> Span<J>
    where
        F: FnMut(I) -> J,
        J: Ord,
    {
        Span::new(f(self.start), f(self.end))
    }
}
impl<I> Span<I> {
    /// Get the start index
    #[inline]
    pub fn start(self) -> I {
        self.start
    }

    /// Get the end index
    #[inline]
    pub fn end(self) -> I {
        self.end
    }
}
impl<I: Index> Span<I> {
    /// Makes a span from offsets relative to the start of this span.
    #[inline]
    pub fn subspan(&self, begin: I::Offset, end: I::Offset) -> Span<I> {
        debug_assert!(end >= begin);
        debug_assert!(self.start() + end <= self.end());
        Span {
            start: self.start() + begin,
            end: self.start() + end,
        }
    }

    /// Create a new span from this one by shrinking the front of the span, i.e. increasing it by
    /// some offset
    #[inline]
    pub fn shrink_front(self, shift: I::Offset) -> Span<I> {
        Span::new(self.start() + shift, self.end())
    }

    /// Create a new span from a byte start and an offset
    #[inline]
    pub fn from_offset(start: I, off: I::Offset) -> Span<I> {
        Span::new(start, start + off)
    }

    /// Return a new span with the low byte position replaced with the supplied byte position
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let span = Span::new(ByteIndex(3), ByteIndex(6));
    /// assert_eq!(
    ///     span.with_start(ByteIndex(2)),
    ///     Span::new(ByteIndex(2), ByteIndex(6))
    /// );
    /// assert_eq!(
    ///     span.with_start(ByteIndex(5)),
    ///     Span::new(ByteIndex(5), ByteIndex(6))
    /// );
    /// assert_eq!(
    ///     span.with_start(ByteIndex(7)),
    ///     Span::new(ByteIndex(6), ByteIndex(7))
    /// );
    /// ```
    #[inline]
    pub fn with_start(self, start: I) -> Span<I> {
        Span::new(start, self.end())
    }

    /// Return a new span with the high byte position replaced with the supplied byte position
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let span = Span::new(ByteIndex(3), ByteIndex(6));
    /// assert_eq!(
    ///     span.with_end(ByteIndex(7)),
    ///     Span::new(ByteIndex(3), ByteIndex(7))
    /// );
    /// assert_eq!(
    ///     span.with_end(ByteIndex(5)),
    ///     Span::new(ByteIndex(3), ByteIndex(5))
    /// );
    /// assert_eq!(
    ///     span.with_end(ByteIndex(2)),
    ///     Span::new(ByteIndex(2), ByteIndex(3))
    /// );
    /// ```
    #[inline]
    pub fn with_end(self, end: I) -> Span<I> {
        Span::new(self.start(), end)
    }

    /// Return true if `self` fully encloses `other`.
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let a = Span::new(ByteIndex(5), ByteIndex(8));
    ///
    /// assert_eq!(a.contains(a), true);
    /// assert_eq!(a.contains(Span::new(ByteIndex(6), ByteIndex(7))), true);
    /// assert_eq!(a.contains(Span::new(ByteIndex(6), ByteIndex(10))), false);
    /// assert_eq!(a.contains(Span::new(ByteIndex(3), ByteIndex(6))), false);
    /// ```
    #[inline]
    pub fn contains(self, other: Span<I>) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }

    /// Return `Equal` if `self` contains `pos`, otherwise it returns `Less` if `pos` is before
    /// `start` or `Greater` if `pos` is after or at `end`.
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    /// use std::cmp::Ordering::*;
    ///
    /// let a = Span::new(ByteIndex(5), ByteIndex(8));
    ///
    /// assert_eq!(a.containment(ByteIndex(4)), Less);
    /// assert_eq!(a.containment(ByteIndex(5)), Equal);
    /// assert_eq!(a.containment(ByteIndex(6)), Equal);
    /// assert_eq!(a.containment(ByteIndex(8)), Equal);
    /// assert_eq!(a.containment(ByteIndex(9)), Greater);
    /// ```
    #[inline]
    pub fn containment(self, pos: I) -> Ordering {
        use std::cmp::Ordering::*;

        match (pos.cmp(&self.start), pos.cmp(&self.end)) {
            (Equal, _) | (_, Equal) | (Greater, Less) => Equal,
            (Less, _) => Less,
            (_, Greater) => Greater,
        }
    }

    /// Return `Equal` if `self` contains `pos`, otherwise it returns `Less` if `pos` is before
    /// `start` or `Greater` if `pos` is *strictly* after `end`.
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    /// use std::cmp::Ordering::*;
    ///
    /// let a = Span::new(ByteIndex(5), ByteIndex(8));
    ///
    /// assert_eq!(a.containment_exclusive(ByteIndex(4)), Less);
    /// assert_eq!(a.containment_exclusive(ByteIndex(5)), Equal);
    /// assert_eq!(a.containment_exclusive(ByteIndex(6)), Equal);
    /// assert_eq!(a.containment_exclusive(ByteIndex(8)), Greater);
    /// assert_eq!(a.containment_exclusive(ByteIndex(9)), Greater);
    /// ```
    #[inline]
    pub fn containment_exclusive(self, pos: I) -> Ordering {
        if self.end == pos {
            Ordering::Greater
        } else {
            self.containment(pos)
        }
    }

    /// Return a `Span` that would enclose both `self` and `end`.
    ///
    /// ```plain
    /// self     ~~~~~~~
    /// end                     ~~~~~~~~
    /// returns  ~~~~~~~~~~~~~~~~~~~~~~~
    /// ```
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let a = Span::new(ByteIndex(2), ByteIndex(5));
    /// let b = Span::new(ByteIndex(10), ByteIndex(14));
    ///
    /// assert_eq!(a.to(b), Span::new(ByteIndex(2), ByteIndex(14)));
    /// ```
    #[inline]
    pub fn to(self, end: Span<I>) -> Span<I> {
        Span::new(
            cmp::min(self.start(), end.start()),
            cmp::max(self.end(), end.end()),
        )
    }

    /// Return a `Span` between the end of `self` to the beginning of `end`.
    ///
    /// ```plain
    /// self     ~~~~~~~
    /// end                     ~~~~~~~~
    /// returns         ~~~~~~~~~
    /// ```
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let a = Span::new(ByteIndex(2), ByteIndex(5));
    /// let b = Span::new(ByteIndex(10), ByteIndex(14));
    ///
    /// assert_eq!(a.between(b), Span::new(ByteIndex(5), ByteIndex(10)));
    /// ```
    #[inline]
    pub fn between(self, end: Span<I>) -> Span<I> {
        Span::new(self.end(), end.start())
    }

    /// Return a `Span` between the beginning of `self` to the beginning of `end`.
    ///
    /// ```plain
    /// self     ~~~~~~~
    /// end                     ~~~~~~~~
    /// returns  ~~~~~~~~~~~~~~~~
    /// ```
    ///
    /// ```rust
    /// use liblumen_diagnostics::{ByteIndex, Span};
    ///
    /// let a = Span::new(ByteIndex(2), ByteIndex(5));
    /// let b = Span::new(ByteIndex(10), ByteIndex(14));
    ///
    /// assert_eq!(a.until(b), Span::new(ByteIndex(2), ByteIndex(10)));
    /// ```
    #[inline]
    pub fn until(self, end: Span<I>) -> Span<I> {
        Span::new(self.start(), end.start())
    }
}

impl<I: fmt::Display> fmt::Display for Span<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.start.fmt(f)?;
        write!(f, "..")?;
        self.end.fmt(f)?;
        Ok(())
    }
}

/// A span of byte indices
pub type ByteSpan = Span<ByteIndex>;

pub const DUMMY_SPAN: Span<ByteIndex> = Span {
    start: ByteIndex(0),
    end: ByteIndex(0),
};
