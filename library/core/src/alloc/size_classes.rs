use core::fmt::{self, Debug};
use core::mem;

/// A wrapper around `usize` values which define a size class
/// in terms of its size in words
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SizeClass(usize);

impl SizeClass {
    #[inline]
    pub const fn new(words: usize) -> Self {
        Self(words)
    }

    #[inline]
    pub fn to_bytes(&self) -> usize {
        self.0 * mem::size_of::<usize>()
    }

    #[inline]
    pub fn as_words(&self) -> usize {
        self.0
    }

    #[inline]
    pub fn round_to_nearest_word(&self) -> Self {
        Self(next_factor_of_word(self.0))
    }
}

impl Debug for SizeClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SizeClass")
            .field("byte_len", &self.to_bytes())
            .field("word_len", &self.as_words())
            .finish()
    }
}

/// Represents a type which can map allocation sizes to size class sizes
pub trait SizeClassIndex {
    /// Given a SizeClass returned by `size_class_for`, this returns the
    /// position of the size class in the index
    fn index_for(&self, size_class: SizeClass) -> usize;

    /// Maps a requested allocation size to the nearest size class size,
    /// if a size class is available to fill the request, otherwise returns None
    fn size_class_for(&self, request_size: usize) -> Option<SizeClass>;

    /// Same as size_class for, but optimized when the request size is known to be valid
    unsafe fn size_class_for_unchecked(&self, request_size: usize) -> SizeClass;
}

/// Calculates the next nearest factor of the target word size fits `n`
#[inline]
pub fn next_factor_of_word(n: usize) -> usize {
    let base = n / mem::size_of::<usize>();
    let rem = n % mem::size_of::<usize>();
    if rem == 0 {
        base
    } else {
        base + 1
    }
}
