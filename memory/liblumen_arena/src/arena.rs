//! NOTE: Modified version of impl in rustc
//!
//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.
//!
//! This crate implements `TypedArena`, a simple arena that can only hold
//! objects of a single type.
use core::cell::{Cell, RefCell};
use core::cmp;
use core::intrinsics;
use core::marker::{PhantomData, Send};
use core::mem;
use core::ptr;
use core::slice;

use alloc::raw_vec::RawVec;
use alloc::vec::Vec;

/// An arena that can hold objects of only one type.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut T>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<TypedArenaChunk<T>>>,

    /// Marker indicating that dropping the arena causes its owned
    /// instances of `T` to be dropped.
    _own: PhantomData<T>,
}

struct TypedArenaChunk<T> {
    /// The raw storage for the arena chunk.
    storage: RawVec<T>,
}

impl<T> TypedArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> TypedArenaChunk<T> {
        TypedArenaChunk {
            storage: RawVec::with_capacity(capacity),
        }
    }

    /// Destroys this arena chunk.
    #[inline]
    unsafe fn destroy(&mut self, len: usize) {
        // The branch on needs_drop() is an -O1 performance optimization.
        // Without the branch, dropping TypedArena<u8> takes linear time.
        if mem::needs_drop::<T>() {
            let mut start = self.start();
            // Destroy all allocated objects.
            for _ in 0..len {
                ptr::drop_in_place(start);
                start = start.offset(1);
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&self) -> *mut T {
        self.storage.ptr()
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&self) -> *mut T {
        unsafe {
            if mem::size_of::<T>() == 0 {
                // A pointer as large as possible for zero-sized elements.
                !0 as *mut T
            } else {
                self.start().add(self.storage.capacity())
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
const PAGE: usize = 4 * 1024;

#[cfg(target_arch = "wasm32")]
const PAGE: usize = 64 * 1024;

impl<T> Default for TypedArena<T> {
    /// Creates a new `TypedArena`.
    fn default() -> TypedArena<T> {
        TypedArena {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(0 as *mut T),
            end: Cell::new(0 as *mut T),
            chunks: RefCell::new(Vec::new()),
            _own: PhantomData,
        }
    }
}

impl<T> TypedArena<T> {
    pub fn in_arena(&self, ptr: *const T) -> bool {
        let ptr = ptr as *const T as *mut T;

        self.chunks
            .borrow()
            .iter()
            .any(|chunk| chunk.start() <= ptr && ptr < chunk.end())
    }
    /// Allocates an object in the `TypedArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        if self.ptr == self.end {
            self.grow(1)
        }

        unsafe {
            if mem::size_of::<T>() == 0 {
                self.ptr
                    .set(intrinsics::arith_offset(self.ptr.get() as *mut u8, 1) as *mut T);
                let ptr = mem::align_of::<T>() as *mut T;
                // Don't drop the object. This `write` is equivalent to `forget`.
                ptr::write(ptr, object);
                &mut *ptr
            } else {
                let ptr = self.ptr.get();
                // Advance the pointer.
                self.ptr.set(self.ptr.get().offset(1));
                // Write into uninitialized memory.
                ptr::write(ptr, object);
                &mut *ptr
            }
        }
    }

    /// Allocates a slice of objects that are copied into the `TypedArena`, returning a mutable
    /// reference to it. Will panic if passed a zero-sized types.
    ///
    /// Panics:
    ///
    ///  - Zero-sized types
    ///  - Zero-length slices
    #[inline]
    pub fn alloc_slice(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        assert!(mem::size_of::<T>() != 0);
        assert!(slice.len() != 0);

        let available_capacity_bytes = self.end.get() as usize - self.ptr.get() as usize;
        let at_least_bytes = slice.len() * mem::size_of::<T>();
        if available_capacity_bytes < at_least_bytes {
            self.grow(slice.len());
        }

        unsafe {
            let start_ptr = self.ptr.get();
            let arena_slice = slice::from_raw_parts_mut(start_ptr, slice.len());
            self.ptr.set(start_ptr.add(arena_slice.len()));
            arena_slice.copy_from_slice(slice);
            arena_slice
        }
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self, n: usize) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let (chunk, mut new_capacity);
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                let currently_used_cap = used_bytes / mem::size_of::<T>();
                if last_chunk.storage.reserve_in_place(currently_used_cap, n) {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    new_capacity = last_chunk.storage.capacity();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= currently_used_cap + n {
                            break;
                        }
                    }
                }
            } else {
                let elem_size = cmp::max(1, mem::size_of::<T>());
                new_capacity = cmp::max(n, PAGE / elem_size);
            }
            chunk = TypedArenaChunk::<T>::new(new_capacity);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        unsafe {
            // Clear the last chunk, which is partially filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            if let Some(mut last_chunk) = chunks_borrow.last_mut() {
                self.clear_last_chunk(&mut last_chunk);
                let len = chunks_borrow.len();
                // If `T` is ZST, code below has no effect.
                for mut chunk in chunks_borrow.drain(..len - 1) {
                    let cap = chunk.storage.capacity();
                    chunk.destroy(cap);
                }
            }
        }
    }

    // Drops the contents of the last chunk. The last chunk is partially empty, unlike all other
    // chunks.
    fn clear_last_chunk(&self, last_chunk: &mut TypedArenaChunk<T>) {
        // Determine how much was filled.
        let start = last_chunk.start() as usize;
        // We obtain the value of the pointer to the first uninitialized element.
        let end = self.ptr.get() as usize;
        // We then calculate the number of elements to be dropped in the last chunk,
        // which is the filled area's length.
        let diff = if mem::size_of::<T>() == 0 {
            // `T` is ZST. It can't have a drop flag, so the value here doesn't matter. We get
            // the number of zero-sized values in the last and only chunk, just out of caution.
            // Recall that `end` was incremented for each allocated value.
            end - start
        } else {
            (end - start) / mem::size_of::<T>()
        };
        // Pass that to the `destroy` method.
        unsafe {
            last_chunk.destroy(diff);
        }
        // Reset the chunk.
        self.ptr.set(last_chunk.start());
    }
}

unsafe impl<#[may_dangle] T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        unsafe {
            // Determine how much was filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            if let Some(mut last_chunk) = chunks_borrow.pop() {
                // Drop the contents of the last chunk.
                self.clear_last_chunk(&mut last_chunk);
                // The last chunk will be dropped. Destroy all other chunks.
                for chunk in chunks_borrow.iter_mut() {
                    let cap = chunk.storage.capacity();
                    chunk.destroy(cap);
                }
            }
            // RawVec handles deallocation of `last_chunk` and `self.chunks`.
        }
    }
}

unsafe impl<T: Send> Send for TypedArena<T> {}

pub struct DroplessArena {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut u8>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut u8>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<TypedArenaChunk<u8>>>,
}

unsafe impl Send for DroplessArena {}

impl Default for DroplessArena {
    #[inline]
    fn default() -> DroplessArena {
        DroplessArena {
            ptr: Cell::new(0 as *mut u8),
            end: Cell::new(0 as *mut u8),
            chunks: Default::default(),
        }
    }
}

impl DroplessArena {
    pub fn in_arena<T: ?Sized>(&self, ptr: *const T) -> bool {
        let ptr = ptr as *const u8 as *mut u8;

        self.chunks
            .borrow()
            .iter()
            .any(|chunk| chunk.start() <= ptr && ptr < chunk.end())
    }

    #[inline]
    fn align(&self, align: usize) {
        let final_address = ((self.ptr.get() as usize) + align - 1) & !(align - 1);
        self.ptr.set(final_address as *mut u8);
        assert!(self.ptr <= self.end);
    }

    #[inline(never)]
    #[cold]
    fn grow(&self, needed_bytes: usize) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let (chunk, mut new_capacity);
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                if last_chunk
                    .storage
                    .reserve_in_place(used_bytes, needed_bytes)
                {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    new_capacity = last_chunk.storage.capacity();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= used_bytes + needed_bytes {
                            break;
                        }
                    }
                }
            } else {
                new_capacity = cmp::max(needed_bytes, PAGE);
            }
            chunk = TypedArenaChunk::<u8>::new(new_capacity);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    #[inline]
    pub unsafe fn alloc_raw(&self, bytes: usize, align: usize) -> *mut u8 {
        assert!(bytes != 0);

        self.align(align);

        let future_end = intrinsics::arith_offset(self.ptr.get(), bytes as isize);
        if (future_end as *mut u8) >= self.end.get() {
            self.grow(bytes);
        }

        let ptr = self.ptr.get();
        // Set the pointer past ourselves
        self.ptr
            .set(intrinsics::arith_offset(self.ptr.get(), bytes as isize) as *mut u8);

        ptr
    }

    #[inline]
    pub fn alloc_copy<T>(&self, object: T) -> &mut T {
        assert!(!mem::needs_drop::<T>());

        unsafe {
            let mem = self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) as *mut T;

            // Write into uninitialized memory.
            ptr::write(mem, object);
            &mut *mem
        }
    }

    /// Allocates a slice of objects that are copied into the `DroplessArena`, returning a mutable
    /// reference to it. Will panic if passed a zero-sized type.
    ///
    /// Panics:
    ///
    ///  - Zero-sized types
    ///  - Zero-length slices
    #[inline]
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(!slice.is_empty());

        unsafe {
            let mem = self.alloc_raw(slice.len() * mem::size_of::<T>(), mem::align_of::<T>())
                as *mut _ as *mut T;

            let arena_slice = slice::from_raw_parts_mut(mem, slice.len());
            arena_slice.copy_from_slice(slice);
            arena_slice
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TypedArena;
    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::Cell;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use test::Bencher;

    #[allow(dead_code)]
    #[derive(Debug, Eq, PartialEq)]
    struct Point {
        x: i32,
        y: i32,
        z: i32,
    }

    #[test]
    pub fn test_unused() {
        let arena: TypedArena<Point> = TypedArena::default();
        assert!(arena.chunks.borrow().is_empty());
    }

    #[test]
    fn test_arena_alloc_nested() {
        struct Inner {
            value: u8,
        }
        struct Outer<'a> {
            inner: &'a Inner,
        }
        enum EI<'e> {
            I(Inner),
            O(Outer<'e>),
        }

        struct Wrap<'a>(TypedArena<EI<'a>>);

        impl<'a> Wrap<'a> {
            fn alloc_inner<F: Fn() -> Inner>(&self, f: F) -> &Inner {
                let r: &EI = self.0.alloc(EI::I(f()));
                if let &EI::I(ref i) = r {
                    i
                } else {
                    panic!("mismatch");
                }
            }
            fn alloc_outer<F: Fn() -> Outer<'a>>(&self, f: F) -> &Outer {
                let r: &EI = self.0.alloc(EI::O(f()));
                if let &EI::O(ref o) = r {
                    o
                } else {
                    panic!("mismatch");
                }
            }
        }

        let arena = Wrap(TypedArena::default());

        let result = arena.alloc_outer(|| Outer {
            inner: arena.alloc_inner(|| Inner { value: 10 }),
        });

        assert_eq!(result.inner.value, 10);
    }

    #[test]
    pub fn test_copy() {
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
        }
    }

    #[bench]
    pub fn bench_copy(b: &mut Bencher) {
        let arena = TypedArena::default();
        b.iter(|| arena.alloc(Point { x: 1, y: 2, z: 3 }))
    }

    #[bench]
    pub fn bench_copy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = Box::new(Point { x: 1, y: 2, z: 3 });
        })
    }

    #[allow(dead_code)]
    struct Noncopy {
        string: String,
        array: Vec<i32>,
    }

    #[test]
    pub fn test_noncopy() {
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(Noncopy {
                string: String::from("hello world"),
                array: vec![1, 2, 3, 4, 5],
            });
        }
    }

    #[test]
    pub fn test_typed_arena_zero_sized() {
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(());
        }
    }

    #[test]
    pub fn test_typed_arena_clear() {
        let mut arena = TypedArena::default();
        for _ in 0..10 {
            arena.clear();
            for _ in 0..10000 {
                arena.alloc(Point { x: 1, y: 2, z: 3 });
            }
        }
    }

    #[bench]
    pub fn bench_typed_arena_clear(b: &mut Bencher) {
        let mut arena = TypedArena::default();
        b.iter(|| {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
            arena.clear();
        })
    }

    // Drop tests

    struct DropCounter<'a> {
        count: &'a Cell<u32>,
    }

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.count.set(self.count.get() + 1);
        }
    }

    #[test]
    fn test_typed_arena_drop_count() {
        let counter = Cell::new(0);
        {
            let arena: TypedArena<DropCounter> = TypedArena::default();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(DropCounter { count: &counter });
            }
        };
        assert_eq!(counter.get(), 100);
    }

    #[test]
    fn test_typed_arena_drop_on_clear() {
        let counter = Cell::new(0);
        let mut arena: TypedArena<DropCounter> = TypedArena::default();
        for i in 0..10 {
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(DropCounter { count: &counter });
            }
            arena.clear();
            assert_eq!(counter.get(), i * 100 + 100);
        }
    }

    static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct SmallDroppable;

    impl Drop for SmallDroppable {
        fn drop(&mut self) {
            DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_typed_arena_drop_small_count() {
        DROP_COUNTER.store(0, Ordering::SeqCst);
        {
            let arena: TypedArena<SmallDroppable> = TypedArena::default();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(SmallDroppable);
            }
            // dropping
        };
        assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 100);
    }

    #[bench]
    pub fn bench_noncopy(b: &mut Bencher) {
        let arena = TypedArena::default();
        b.iter(|| {
            arena.alloc(Noncopy {
                string: String::from("hello world"),
                array: vec![1, 2, 3, 4, 5],
            })
        })
    }

    #[bench]
    pub fn bench_noncopy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = Box::new(Noncopy {
                string: String::from("hello world"),
                array: vec![1, 2, 3, 4, 5],
            });
        })
    }
}
