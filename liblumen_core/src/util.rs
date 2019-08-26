///! This module/namespace contains a variety of helpful utility
///! functions and types which are used throughout Lumen
pub mod cache_padded;
pub mod pointer;
pub mod reference;

#[macro_export]
macro_rules! offset_of {
    ($strukt:path, $field:ident) => ({
        // Using a separate function to minimize unhygienic hazards
        // (e.g. unsafety of #[repr(packed)] field borrows).
        // Uncomment `const` when `const fn`s can juggle pointers.
        fn offset() -> usize {
            let u = core::mem::MaybeUninit::<$strukt>::uninit();
            // Use pattern-matching to avoid accidentally going through Deref.
            let &$strukt { $field: ref f, .. } = unsafe { &*u.as_ptr() };
            let o = (f as *const _ as usize).wrapping_sub(&u as *const _ as usize);
            // Triple check that we are within `u` still.
            assert!((0..=core::mem::size_of_val(&u)).contains(&o));
            o
        }
        offset()
    })
}
