/// This struct represents a bounded memory region and is
/// used to determine membership of a pointer within that
/// region
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    start: *const u8,
    end: *const u8,
}
impl Region {
    /// Creates a `Region` given start/end pointers
    ///
    /// Panics if the region is zero-sized or if the
    /// start and end pointers are swapped
    pub fn new<T, U>(start: *const T, end: *const U) -> Self
    where
        T: ?Sized,
        U: ?Sized,
    {
        let start = start as *const u8;
        let end = end as *const u8;
        assert!(
            start < end,
            "invalid region bounds, `start` must have a lower address than `end`"
        );
        assert!(
            end as usize - start as usize > 0,
            "invalid region bounds, must represent at least one byte"
        );
        Self { start, end }
    }

    /// Returns the size of this region in bytes
    pub fn size(&self) -> usize {
        use crate::util::pointer::distance_absolute;

        distance_absolute(self.end, self.start)
    }

    /// Returns true if `ptr` is contained in this region
    #[inline]
    pub fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        use crate::util::pointer::in_area;

        in_area(ptr, self.start, self.end)
    }
}
