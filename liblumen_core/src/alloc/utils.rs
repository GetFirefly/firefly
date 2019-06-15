///! Contains utility and shared functionality used as part of the overall
///! allocator framework provided by this crate.
use core::mem;

use core_alloc::string::String;

/// Used to assert alignment against the size of the target pointer width
#[allow(unused_macros)]
#[macro_export]
macro_rules! assert_word_aligned {
    ($ptr:expr) => {
        assert_aligned_to!($ptr, core::mem::size_of::<usize>())
    };
}

/// Used to assert alignment against an arbitrary size
#[allow(unused_macros)]
#[macro_export]
macro_rules! assert_aligned_to {
    ($ptr:expr, $align:expr) => {
        assert!(
            liblumen_core::alloc::alloc_utils::is_aligned_at($ptr, $align),
            "{:p} is not aligned to {}",
            $ptr,
            $align
        )
    };
}

// A default good size allocation is deduced as the size of `T` rounded up
// to the required alignment of `T`
#[inline(always)]
pub fn good_alloc_size<T>() -> usize {
    // TODO: Need to factor in allocator min alignment
    self::round_up_to_multiple_of(mem::size_of::<T>(), mem::align_of::<T>())
}

/// Like regular division, but rounds up
#[inline(always)]
pub fn divide_round_up(x: usize, y: usize) -> usize {
    assert!(y > 0);
    (x + y - 1) / y
}

// Returns `size` rounded up to a multiple of `base`.
#[inline]
pub fn round_up_to_multiple_of(size: usize, base: usize) -> usize {
    let rem = size % base;
    if rem == 0 {
        size
    } else {
        size + base - rem
    }
}

// Rounds up `size` to a multiple of `align`, which must be a power of two
#[inline(always)]
pub fn round_up_to_alignment(size: usize, align: usize) -> usize {
    assert!(align.is_power_of_two());
    self::round_up_to_multiple_of(size, align)
}

// Rounds down `size` to a multiple of `align`, which must be a power of two
#[inline(always)]
pub fn round_down_to_alignment(size: usize, align: usize) -> usize {
    assert!(align.is_power_of_two());
    // This trick works by masking the low bits in `size`
    // up to (but not including) `align`, the result is
    // the value of `size` rounded down to the next nearest
    // number which is aligned to `align`
    //
    // EXAMPLE: given `size = 1048` and `align = 1024`, the result
    // is `1024`, which can be seen with the following 16bit representation:
    //
    //     size:                0000010000011000
    //     align - 1:           0000001111111111
    //     !align - 1:          1111110000000000
    //     size & !(align - 1): 0000010000000000
    size & !(align - 1)
}

// Shifts the given pointer up to the next nearest aligned byte
#[inline(always)]
pub fn align_up_to(ptr: *const u8, align: usize) -> *const u8 {
    self::round_up_to_alignment(ptr as usize, align) as *const u8
}

// Aligns the given pointer down to the given alignment.
// The resulting pointer is either less than or equal to the given pointer.
#[inline(always)]
pub fn align_down_to(ptr: *const u8, align: usize) -> *const u8 {
    assert!(align.is_power_of_two());
    (ptr as usize & !(align - 1)) as *const u8
}

// Aligns the given pointer up to the next nearest byte which is a multiple of `base`
#[inline(always)]
pub fn align_up_to_multiple_of(ptr: *const u8, base: usize) -> *const u8 {
    self::round_up_to_multiple_of(ptr as usize, base) as *const u8
}

// Returns true if `ptr` is aligned to `align`
#[inline(always)]
pub fn is_aligned_at<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize) % align == 0
}

// Returns the effective alignment of `ptr`, i.e. the largest power
// of two that is a divisor of `ptr`
#[inline(always)]
pub fn effective_alignment(ptr: *const u8) -> usize {
    1usize << (ptr as usize).trailing_zeros()
}

/// Given a reference to an object, formats the reference
/// as a hexadecimal memory address
#[cfg(target_pointer_width = "32")]
#[inline(always)]
pub fn format_address_of<T>(ptr: &T) -> String {
    format!("{:#x}", ptr as *const _ as *const u8 as usize)
}

/// Given a reference to an object, formats the reference
/// as a hexadecimal memory address
#[cfg(target_pointer_width = "64")]
#[inline(always)]
pub fn format_address_of<T>(ptr: &T) -> String {
    format!("{:#x}", ptr as *const _ as *const u8 as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;

    #[allow(dead_code)]
    pub struct Example {
        ptr: *const u8,
        refc: AtomicUsize,
    }

    #[test]
    fn good_alloc_size_test() {
        assert_eq!(good_alloc_size::<usize>(), 8);
        assert_eq!(good_alloc_size::<Cell<usize>>(), 8);
        assert_eq!(good_alloc_size::<Example>(), 16);
    }

    #[test]
    fn round_up_to_multiple_of_test() {
        assert_eq!(round_up_to_multiple_of(10, 11), 11);
        assert_eq!(round_up_to_multiple_of(11, 11), 11);
        assert_eq!(round_up_to_multiple_of(12, 11), 22);
        assert_eq!(round_up_to_multiple_of(118, 11), 121);
    }

    #[test]
    fn round_up_to_alignment_test() {
        assert_eq!(round_up_to_alignment(10, 4), 12);
        assert_eq!(round_up_to_alignment(11, 2), 12);
        assert_eq!(round_up_to_alignment(12, 8), 16);
        assert_eq!(round_up_to_alignment(118, 64), 128);
    }

    #[test]
    fn round_down_to_alignment_test() {
        assert_eq!(round_down_to_alignment(10, 4), 8);
        assert_eq!(round_down_to_alignment(11, 2), 10);
        assert_eq!(round_down_to_alignment(12, 8), 8);
        assert_eq!(round_down_to_alignment(63, 64), 0);
    }

    #[test]
    fn align_up_to_test() {
        let x: usize = 8;
        let y: *const u8 = 16usize as *const u8;
        assert_eq!(align_up_to(x as *const u8, 16), y);

        let x: usize = 16;
        let y: *const u8 = 16usize as *const u8;
        assert_eq!(align_up_to(x as *const u8, 16), y);
    }

    #[test]
    fn align_down_to_test() {
        let x: usize = 16;
        let y: *const u8 = 16usize as *const u8;
        assert_eq!(align_down_to(x as *const u8, 16), y);

        let x: usize = 14;
        let y: *const u8 = 8usize as *const u8;
        assert_eq!(align_down_to(x as *const u8, 8), y);
    }

    #[test]
    fn align_up_to_multiple_of_test() {
        let x: usize = 8;
        let y: *const u8 = 4096usize as *const u8;
        assert_eq!(align_up_to_multiple_of(x as *const u8, 4096), y);
    }

    #[test]
    fn is_aligned_at_test() {
        let x: usize = 4096;
        assert!(is_aligned_at(x as *const u8, 8));
        assert!(is_aligned_at(x as *const u8, 16));
        assert!(is_aligned_at(x as *const u8, 4096));
        let y: usize = 4092;
        assert!(is_aligned_at(y as *const u8, 4));
        assert!(!is_aligned_at(y as *const u8, 8));
    }

    #[test]
    fn effective_alignment_test() {
        // This is a real address gathered by testing, should be word-aligned
        let ptr = 0x70000cf815a8usize as *const u8;
        let effective = effective_alignment(ptr);
        assert!(effective.is_power_of_two());
        assert_eq!(effective, mem::align_of::<usize>());

        // This address is super-aligned size * 400000001
        // to give us an address in a "normal" range
        let ptr = 0x5f5e10040000usize as *const u8;
        let effective = effective_alignment(ptr);
        assert!(effective.is_power_of_two());
        assert_eq!(effective, 262144);

        let max = 1usize << mem::size_of::<usize>() * 8 - 1;
        let effective = effective_alignment(max as *const u8);
        assert_eq!(effective, max);
    }
}
