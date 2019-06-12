///! This module contains helper functions for working with raw pointers
use core::mem;

/// Returns the relative distance in units of size `T` between `a` and `b`
///
/// If `a` is a higher address than, or equal to `b`, the result is non-negative,
/// otherwise the result is negative.
#[inline]
pub fn distance<T: Sized>(a: *const T, b: *const T) -> isize {
    let offset_bytes = unsafe { (a as *const u8).offset_from(b as *const u8) };
    offset_bytes / mem::size_of::<T>() as isize
}

/// Returns the absolute distance in units of size `T` between `a` and `b`
///
/// Regardless of the order of the arguments, the value will always be non-negative
#[inline(always)]
pub fn distance_absolute<T: Sized>(a: *const T, b: *const T) -> usize {
    unsafe { a.offset_from(b).abs() as usize }
}

/// Returns true if `ptr` is in the memory region between `start` and `end`
///
/// NOTE: If any of the given pointers are null, then false will be returned
#[inline]
pub fn in_area<T, U>(ptr: *const T, start: *const U, end: *const U) -> bool {
    // If any pointers are null, the only sensible answer is false
    if ptr.is_null() || start.is_null() || end.is_null() {
        false
    } else {
        debug_assert!(start as usize <= end as usize);

        let start = start as usize;
        let end = end as usize;
        let ptr = ptr as usize;
        start <= ptr && ptr <= end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        // The distance here is 3 because we're 1 byte into a usize,
        // leaving the distance no longer evenly divisible 4 usizes
        let a = 1 as *const usize;
        let b = (4 * mem::size_of::<usize>()) as *const usize;
        assert_eq!(distance(a, b), -3);
        assert_eq!(distance(b, a), 3);

        // With a unit of bytes, we can see this more clearly, as the size
        // is 1 byte less than 4 usizes in both directions
        let c = a as *const u8;
        let d = b as *const u8;
        assert_eq!(distance(c, d), -4 * (mem::size_of::<usize>() as isize) + 1);
        assert_eq!(distance(d, c), 4 * (mem::size_of::<usize>() as isize) - 1);

        let x = 0 as *const usize;
        let y = 0 as *const usize;
        assert_eq!(distance(x, y), 0);
        assert_eq!(distance(y, x), 0);
    }

    #[test]
    fn test_distance_absolute() {
        let a = 1 as *const usize;
        let b = (4 * mem::size_of::<usize>()) as *const usize;
        assert_eq!(distance_absolute(a, b), 3);
        assert_eq!(distance_absolute(b, a), 3);

        let c = a as *const u8;
        let d = b as *const u8;
        assert_eq!(distance_absolute(c, d), 4 * mem::size_of::<usize>() - 1);
        assert_eq!(distance_absolute(d, c), 4 * mem::size_of::<usize>() - 1);

        let x = 0 as *const usize;
        let y = 0 as *const usize;
        assert_eq!(distance_absolute(x, y), 0);
        assert_eq!(distance_absolute(y, x), 0);
    }

    #[test]
    fn test_in_area() {
        let start = 0 as *const u8;
        let end = unsafe { start.offset(100) };

        assert!(in_area(0 as *const u8, start, end));
        assert!(in_area(100 as *const u8, start, end));
        assert!(!in_area(101 as *const u8, start, end));
    }
}
