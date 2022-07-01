use std::convert::{AsMut, AsRef};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, Range};

use codespan::{ByteIndex, ByteOffset, Span};

use super::{CodeMap, SourceId, SourceIndex};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceSpan {
    pub(crate) source_id: SourceId,
    pub(crate) start: ByteIndex,
    pub(crate) end: ByteIndex,
}
impl Default for SourceSpan {
    #[inline(always)]
    fn default() -> Self {
        Self::UNKNOWN
    }
}
impl SourceSpan {
    /// Represents an invalid/unknown source location
    ///
    /// In most cases we try to use a valid location when generating entities that don't
    /// originate from the original source, i.e. using the module/function span for something
    /// generated at module/function scope. However, in some cases we don't even have the context
    /// to do that, so we use this instead.
    ///
    /// In MLIR this gets lowered as an UnknownLoc, in LLVM it is a no-op
    pub const UNKNOWN: Self = Self {
        source_id: SourceId::UNKNOWN,
        start: ByteIndex(0),
        end: ByteIndex(0),
    };

    /// Creates a new span from `start` to `end`
    ///
    /// This function will panic if the indices are in different source files
    #[inline]
    pub fn new(start: SourceIndex, end: SourceIndex) -> Self {
        let source_id = start.source_id();
        assert_eq!(
            source_id,
            end.source_id(),
            "source spans cannot start and end in different files!"
        );
        let start = start.index();
        let end = end.index();

        Self {
            source_id,
            start,
            end,
        }
    }

    pub fn new_align<F>(
        start: SourceIndex,
        end: SourceIndex,
        get_codemap: &dyn Fn(&mut dyn FnOnce(&CodeMap)),
    ) -> SourceSpan {
        let start_source = start.source_id();
        let end_source = end.source_id();

        if start_source == end_source {
            Self::new(start, end)
        } else {
            let mut result = None;
            get_codemap(&mut |codemap: &CodeMap| {
                let mut idx = start_source;
                loop {
                    if let Some(parent) = codemap.parent(idx) {
                        if idx == end_source {
                            result = Some(Self::new(parent.start(), end));
                            return;
                        }
                        idx = parent.source_id();
                    } else {
                        break;
                    }
                }

                let mut idx = end_source;
                loop {
                    if let Some(parent) = codemap.parent(idx) {
                        if idx == start_source {
                            result = Some(Self::new(start, parent.end()));
                            return;
                        }
                        idx = parent.source_id();
                    } else {
                        break;
                    }
                }
            });
            result.expect("source spans cannot be aligned!")
        }
    }

    /// Returns true if this span represents an "unknown" source span
    #[inline(always)]
    pub fn is_unknown(self) -> bool {
        self == Self::UNKNOWN
    }

    /// Returns the source id associated with this span
    #[inline(always)]
    pub fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Returns the starting source index of this span
    #[inline(always)]
    pub fn start(&self) -> SourceIndex {
        SourceIndex::new(self.source_id, self.start)
    }

    /// Returns the starting byte index of this span in its SourceFile
    #[inline(always)]
    pub fn start_index(&self) -> ByteIndex {
        self.start
    }

    /// Shrinks this span by truncating `offset` bytes from the start of its range
    pub fn shrink_front(mut self, offset: ByteOffset) -> Self {
        self.start += offset;
        self
    }

    /// Returns the ending source index of this span
    #[inline(always)]
    pub fn end(&self) -> SourceIndex {
        SourceIndex::new(self.source_id, self.end)
    }

    /// Returns the ending byte index of this span in its SourceFile
    #[inline(always)]
    pub fn end_index(&self) -> ByteIndex {
        self.end
    }
}
impl Into<Span> for SourceSpan {
    #[inline]
    fn into(self) -> Span {
        Span::new(self.start, self.end)
    }
}
impl Into<ByteIndex> for SourceSpan {
    #[inline]
    fn into(self) -> ByteIndex {
        self.start_index()
    }
}
impl From<SourceSpan> for Range<usize> {
    fn from(span: SourceSpan) -> Range<usize> {
        span.start.into()..span.end.into()
    }
}
impl From<SourceSpan> for Range<SourceIndex> {
    fn from(span: SourceSpan) -> Range<SourceIndex> {
        let start = SourceIndex::new(span.source_id, span.start);
        let end = SourceIndex::new(span.source_id, span.end);
        start..end
    }
}

pub struct Spanned<T: ?Sized> {
    pub span: SourceSpan,
    pub item: T,
}
impl<T> Spanned<T> {
    pub const fn new(span: SourceSpan, item: T) -> Self {
        Self { span, item }
    }
}
impl<T: ?Sized> Spanned<T> {
    /// Return the SourceSpan of this item
    pub const fn span(&self) -> SourceSpan {
        self.span
    }
}
impl<T> From<T> for Spanned<T> {
    fn from(item: T) -> Self {
        Self {
            span: SourceSpan::default(),
            item,
        }
    }
}
impl<T: ?Sized> AsRef<T> for Spanned<T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.item
    }
}
impl<T: ?Sized> AsMut<T> for Spanned<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        &mut self.item
    }
}
impl<T: ?Sized> Deref for Spanned<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.item
    }
}
impl<T: ?Sized> DerefMut for Spanned<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}
impl<T: Default> Default for Spanned<T> {
    fn default() -> Self {
        Self {
            span: SourceSpan::default(),
            item: T::default(),
        }
    }
}
impl<T: Clone> Clone for Spanned<T> {
    fn clone(&self) -> Self {
        Self {
            span: self.span,
            item: self.item.clone(),
        }
    }
}
impl<T: Copy> Copy for Spanned<T> {}
unsafe impl<T: Send> Send for Spanned<T> {}
unsafe impl<T: Sync> Sync for Spanned<T> {}
impl<T, U> PartialEq<Spanned<U>> for Spanned<T>
where
    T: PartialEq<U>,
    U: PartialEq<T>,
{
    fn eq(&self, other: &Spanned<U>) -> bool {
        self.item.eq(&other.item)
    }
}
impl<T: Eq> Eq for Spanned<T> {}
impl<T, U> PartialOrd<Spanned<U>> for Spanned<T>
where
    T: PartialOrd<U>,
    U: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Spanned<U>) -> Option<std::cmp::Ordering> {
        self.item.partial_cmp(&other.item)
    }
}
impl<T: Ord> Ord for Spanned<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.item.cmp(&other.item)
    }
}
impl<T: Hash> Hash for Spanned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.item.hash(state)
    }
}
impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Spanned({:?}, {:?})", &self.span, &self.item)
    }
}
impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.item)
    }
}
