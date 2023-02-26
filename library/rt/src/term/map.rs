use alloc::alloc::{AllocError, Allocator, Global, Layout};
use alloc::boxed::Box;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::intrinsics::unlikely;
use core::mem::{self, MaybeUninit};
use core::ops::Deref;
use core::ptr::{self, NonNull, Pointee};

use firefly_alloc::heap::Heap;

use crate::cmp::ExactEq;
use crate::gc::Gc;

use super::{Boxable, Header, LayoutBuilder, Metadata, OpaqueTerm, Tag, Term};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MapError {
    BadKey,
    BadMap,
    SizeLimit,
    AllocError(AllocError),
}
impl From<AllocError> for MapError {
    fn from(err: AllocError) -> Self {
        Self::AllocError(err)
    }
}

/// TODO: Placeholder until we can unify the two map variants as `Map`
pub type Map = SmallMap;

#[allow(unused)]
pub struct LargeMap;
impl LargeMap {
    #[allow(unused)]
    pub fn from_iter<I, A>(_iter: I, _alloc: &A) -> Result<Gc<Self>, MapError>
    where
        A: ?Sized + Allocator,
        I: Iterator<Item = (OpaqueTerm, OpaqueTerm)> + ExactSizeIterator,
    {
        todo!()
    }
}

pub const SMALL_MAP_LIMIT: usize = mem::size_of::<u64>() * 8;

/// This type of map is used when the number of key/value pairs is below a certain threshold,
/// the `SMALL_MAP_LIMIT`, which is 64 currently.
///
/// This map implementation stores all of the keys and values in a single contiguous array,
/// sorted by the term order of the keys. The storage places all of the keys first, followed
/// by all of the values.
///
/// # Performance
///
/// Searching by key is performed using binary search and is `O(log n)` in the worst case.
///
/// Insertion/update involves an `O(log n)` search to find the insertion point, and a memcpy of the
/// previous map's array into the new map (as well as the allocation of the new map). This
/// a new map on the heap of appropriate size, and to memcpy the previous map's elements into
/// the new map. An optimization exits to avoid allocating a new map if an insertion would have
/// no effect (i.e. the key exists in the map and the value hasn't changed).
///
/// Deletion involves an `O(log n)` search to find the index of the key/value being removed, and
/// then one or more memcpys depending on where the removed element was located. If in the middle
/// of the map, we need three memcpys, one for the prefix of the array, the middle (between the
/// removed key and the removed value), and the suffix.
///
/// Construction from an iterator of unsorted key/value pairs uses [glidesort](https://docs.rs/glidesort/)
/// with an on-stack buffer to allow sorting the input elements without any heap allocations. This
/// is `O(n)` in the best case, and `O(n log n)` in the worst case, and is a stable and deterministic sort.
///
/// Construction from a slice of presorted key/value pairs can be done in linear time.
#[repr(C)]
pub struct SmallMap {
    header: Header,
    bitmap: u64,
    kv: [OpaqueTerm],
}
impl SmallMap {
    /// Create a new empty map on the global heap
    #[inline]
    pub fn new() -> Box<Self> {
        Self::with_capacity(0)
    }

    /// Create a new, empty map with the given allocator
    pub fn new_in<A: ?Sized + Allocator>(alloc: &A) -> Result<Gc<Self>, AllocError> {
        Self::with_capacity_in(0, alloc)
    }

    /// Create a shallow clone of `other` in `alloc`
    ///
    /// This is unlike `clone_to_heap`, in that the elements of the map are not deep-cloned
    /// to the target heap. Callers must keep this in mind.
    pub fn clone_from<A: ?Sized + Allocator>(
        other: &Self,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let mut boxed = Self::with_capacity_nolimit(other.size(), alloc)?;
        boxed.kv.copy_from_slice(&other.kv);
        Ok(boxed)
    }

    /// Create a new map with the given capacity on the global heap
    ///
    /// All elements of the map will be initialized to `OpaqueTerm::NONE`
    pub fn with_capacity(capacity: usize) -> Box<Self> {
        let empty = ptr::from_raw_parts::<Self>(ptr::null(), capacity);
        let layout = unsafe { Layout::for_value_raw(empty) };
        let ptr: NonNull<()> = Global.allocate(layout).unwrap().cast();
        unsafe {
            let ptr = ptr::from_raw_parts_mut::<Self>(ptr.as_ptr(), capacity * 2);
            let mut boxed = Box::from_raw(ptr);
            boxed.bitmap = 0;
            boxed.header = Header::new(
                Tag::Map,
                MapFlags(MapFlags::FLATMAP | (capacity << 2)).pack(),
            );
            boxed
        }
    }

    /// Create a new, uninitialized map with the given capacity and allocator
    pub fn with_capacity_in<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        assert!(capacity <= SMALL_MAP_LIMIT);
        Self::with_capacity_nolimit(capacity, alloc)
    }

    fn with_capacity_nolimit<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let arity = capacity * 2;
        let mut boxed = Gc::<Self>::with_capacity_in(arity, alloc)?;
        boxed.bitmap = 0;
        boxed.header = Header::new(
            Tag::Map,
            MapFlags(MapFlags::FLATMAP | (capacity << 2)).pack(),
        );
        Ok(boxed)
    }

    /// Creates a new `SmallMap` from an iterator of key/value pairs whose order is not defined.
    ///
    /// If you have a set of elements sorted in term order, see `from_sorted_slice`.
    pub fn from_iter<I, A>(mut iter: I, alloc: &A) -> Result<Gc<Self>, MapError>
    where
        A: ?Sized + Allocator,
        I: Iterator<Item = (OpaqueTerm, OpaqueTerm)> + ExactSizeIterator,
    {
        let capacity = iter.len();
        if capacity > SMALL_MAP_LIMIT {
            return Err(MapError::SizeLimit);
        }

        // Allocate a map which will hold the items
        let mut map = Self::with_capacity_in(capacity, alloc)?;
        match capacity {
            0 => Ok(map),
            1 => {
                let (k, v) = iter.next().unwrap();
                map.bitmap = 1 << 63;
                map.kv[0] = k;
                map.kv[1] = v;
                Ok(map)
            }
            _ => {
                map.from_iter_with_sort(iter);
                Ok(map)
            }
        }
    }

    #[inline(never)]
    fn from_iter_with_sort<I>(&mut self, iter: I)
    where
        I: Iterator<Item = (OpaqueTerm, OpaqueTerm)> + ExactSizeIterator,
    {
        // Allocate two scratch buffers:
        //
        // * Used by the sort algorithm for scratch space
        // * Used to hold the items prior to/during the sort
        let mut buffer = MaybeUninit::<(OpaqueTerm, OpaqueTerm)>::uninit_array::<SMALL_MAP_LIMIT>();
        let mut items = MaybeUninit::<(OpaqueTerm, OpaqueTerm)>::uninit_array::<SMALL_MAP_LIMIT>();

        // Prepare the items for sorting
        let mut ptr = MaybeUninit::slice_as_mut_ptr(&mut items);
        let mut capacity = 0;
        for (k, v) in iter {
            unsafe {
                ptr.write((k, v));
                ptr = ptr.add(1);
                capacity += 1;
            }
        }

        // Get the initialized items slice
        let items = unsafe { MaybeUninit::slice_assume_init_mut(&mut items[..capacity]) };
        // Sort the items
        glidesort::sort_with_buffer_by(items, buffer.as_mut_slice(), |(a, _), (b, _)| {
            compare_keys(*a, *b)
        });

        // Write the sorted key/value pairs to the allocated map
        for (i, (k, v)) in items.iter().enumerate() {
            unsafe {
                *self.kv.get_unchecked_mut(i) = *k;
                *self.kv.get_unchecked_mut(capacity + i) = *v;
            }
        }

        self.bitmap = u64::MAX << (64 - capacity);
    }

    /// Creates a new `SmallMap` from a slice of key/value pairs which are sorted in term order by the key
    ///
    /// # SAFETY
    ///
    /// The following constraints are imposed upon the provided slice, and callers must guarantee that the
    /// provided slice meets these restrictions:
    ///
    /// * The slice must be sorted in term order by key, where key is the first element of each tuple
    /// * The slice must contain no duplicate keys
    ///
    /// If any of these constraints are unmet, the behavior of the resulting map is likely to be incorrect
    /// and is undefined at best.
    pub unsafe fn from_sorted_slice<A: ?Sized + Allocator>(
        pairs: &[(OpaqueTerm, OpaqueTerm)],
        alloc: &A,
    ) -> Result<Gc<Self>, MapError> {
        debug_assert!(pairs.is_sorted());
        let capacity = pairs.len();
        if capacity > SMALL_MAP_LIMIT {
            return Err(MapError::SizeLimit);
        }

        let mut map = Self::with_capacity_in(capacity, alloc)?;
        for (i, (k, v)) in pairs.iter().enumerate() {
            unsafe {
                *map.kv.get_unchecked_mut(i) = *k;
                *map.kv.get_unchecked_mut(capacity + i) = *v;
            }
        }
        map.bitmap = u64::MAX << (64 - capacity);
        Ok(map)
    }

    /// Returns the maximum number of keys this map can hold
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.metadata().capacity()
    }

    /// Returns the number of keys inserted in this map
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.bitmap.leading_ones() as usize
    }

    /// Returns true if this map is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Produces an iterator which traverses each key/value pair in this map
    #[inline]
    pub fn iter(&self) -> SmallMapIter<'_> {
        SmallMapIter::new(self)
    }

    /// Returns the underlying buffer of this map as a slice
    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [OpaqueTerm] {
        &mut self.kv
    }

    #[inline]
    pub fn keys(&self) -> &[OpaqueTerm] {
        &self.kv[..self.size()]
    }

    #[inline]
    pub fn keys_mut(&mut self) -> &mut [OpaqueTerm] {
        let size = self.size();
        &mut self.kv[..size]
    }

    #[inline]
    pub fn values(&self) -> &[OpaqueTerm] {
        let capacity = self.capacity();
        let size = self.size();
        &self.kv[capacity..(capacity + size)]
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [OpaqueTerm] {
        let capacity = self.capacity();
        let size = self.size();
        &mut self.kv[capacity..(capacity + size)]
    }

    /// Returns true if this map contains `key`
    pub fn contains_key(&self, key: OpaqueTerm) -> bool {
        find_key(self, key).is_ok()
    }

    /// Returns the value associated with `key` in this map
    pub fn get<K>(&self, key: K) -> Option<OpaqueTerm>
    where
        K: Into<OpaqueTerm>,
    {
        let index = find_key(self, key.into()).ok()?;
        let capacity = self.capacity();
        debug_assert!(index < self.size());
        Some(unsafe { *self.kv.get_unchecked(capacity + index) })
    }

    /// Inserts the given key/value in the map, updating in-place.
    ///
    /// This function ensures that the sort order of the underlying array is maintained.
    pub fn put_mut<K, V>(&mut self, key: K, value: V)
    where
        K: Into<OpaqueTerm>,
        V: Into<OpaqueTerm>,
    {
        self.do_put_mut(key.into(), value.into());
    }

    fn do_put_mut(&mut self, key: OpaqueTerm, value: OpaqueTerm) {
        use core::cmp::Ordering;

        let key = key.into();
        let capacity = self.capacity();
        let capacity_used = self.size();
        if capacity_used < capacity {
            let next_index = capacity_used;
            // If this is the first insert, things are straightforward
            if next_index == 0 {
                self.kv[0] = key;
                self.kv[capacity] = value.into();
                self.bitmap = u64::MAX << (64 - (capacity_used + 1));
                return;
            }
            // We have at least one available slot to insert into, but we
            // have to determine where our key belongs in the map relative
            // to the other keys. Check for the trivial case where the last
            // key is "less than" our insert key, in which case we can simply
            // insert directly into the available empty slot.
            let last_index = next_index - 1;
            match compare_keys(key, self.kv[last_index]) {
                Ordering::Equal => {
                    // By coincidence, our insert key is identical to the last
                    // key in the map, so update that slot instead.
                    self.kv[last_index] = key;
                    self.kv[capacity + last_index] = value.into();
                    return;
                }
                Ordering::Greater => {
                    // Our insert key can take the next available empty slot
                    self.kv[next_index] = key;
                    self.kv[capacity + next_index] = value.into();
                    self.bitmap = u64::MAX << (64 - (capacity_used + 1));
                    return;
                }
                Ordering::Less => {
                    // Our insert key must be inserted somewhere earlier in the map.
                    //
                    // This requires us to find the desired insertion index, and then
                    // right shift all of the elements in the map after it by 1 slot.
                    match self.kv[..last_index].binary_search_by(|probe| compare_keys(*probe, key))
                    {
                        Ok(index) => {
                            // The key already exists in the map, so we can simply replace its value
                            self.kv[capacity + index] = value.into();
                            return;
                        }
                        Err(index) => {
                            // The key doesn't exist, so we have to shift some elements right
                            self.kv[index..capacity].rotate_right(1);
                            self.kv[(capacity + index)..].rotate_right(1);
                            self.kv[index] = key;
                            self.kv[capacity + index] = value.into();
                            self.bitmap = u64::MAX << (64 - (capacity_used + 1));
                            return;
                        }
                    }
                }
            }
        }

        // We have no extra capacity, but check to see if the key exists in the map first
        if let Ok(index) = find_key(self, key) {
            // Yep, key exists, so insert in that location
            self.kv[capacity + index] = value.into();
        } else {
            panic!("insert requires a larger map");
        }
    }

    /// Inserts `value` under `key` in `map`.
    ///
    /// Returns a new map with the newly inserted/updated key/value.
    ///
    /// This operation will insert the key if it doesn't exist, or update the previous value if found.
    pub fn put<K, V, A>(self: Gc<Self>, key: K, value: V, alloc: &A) -> Result<Gc<Self>, AllocError>
    where
        K: Into<OpaqueTerm>,
        V: Into<OpaqueTerm>,
        A: ?Sized + Allocator,
    {
        self.do_put(key.into(), value.into(), alloc)
    }

    fn do_put<A>(
        self: Gc<Self>,
        key: OpaqueTerm,
        value: OpaqueTerm,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError>
    where
        A: ?Sized + Allocator,
    {
        let n = self.size();
        let capacity = self.capacity();

        // If the base map is empty, we can short-circuit everything else and just create a new single-entry map
        if n == 0 {
            let mut new_map = Self::with_capacity_in(1, alloc)?;
            new_map.kv[0] = key;
            new_map.kv[1] = value;
            new_map.bitmap = 1 << 63;
            return Ok(new_map);
        }

        match find_key(&self, key) {
            // We found the key in the old map, the map will remain the same size
            Ok(index) => {
                let value_index = capacity + index;
                debug_assert!(value_index < self.kv.len());
                let old = unsafe { *self.kv.get_unchecked(value_index) };

                // If values are trivially identical, we can reduce this to a pointer copy
                if old == value {
                    return Ok(self);
                }

                // We have to allocate a new map since it has been changed
                let mut new_map = Self::with_capacity_in(n, alloc)?;
                new_map.bitmap = u64::MAX << (64 - n);
                if n == capacity {
                    new_map.kv.copy_from_slice(&self.kv);
                } else {
                    new_map.keys_mut().copy_from_slice(self.keys());
                    new_map.values_mut().copy_from_slice(self.values());
                }
                unsafe {
                    *new_map.kv.get_unchecked_mut(n + index) = value;
                }
                Ok(new_map)
            }
            // We are inserting a new key, and the previous map was already at the boundary for
            // small maps; we have to promote this map to a large map.
            Err(_) if n >= SMALL_MAP_LIMIT => {
                /*
                let iter = core::iter::once((key, value)).chain(
                    self.keys()
                        .iter()
                        .copied()
                        .zip(self.values().iter().copied()),
                );
                return LargeMap::from_iter(iter, alloc);
                */
                unimplemented!()
            }
            // After this, are inserting a new key, but we are still below the small map limit
            Err(ip) => {
                let new_len = n + 1;
                let mut new_map = Self::with_capacity_in(new_len, alloc)?;
                new_map.bitmap = u64::MAX << (64 - new_len);
                if ip == 0 {
                    // The insertion point is at the start of the old map, so insertion is straightforward
                    new_map.kv[1..new_len].copy_from_slice(self.keys());
                    new_map.kv[(new_len + 1)..].copy_from_slice(self.values());
                    // We're inserting into the very first value slot
                    new_map.kv[0] = key;
                    new_map.kv[new_len] = value;
                } else if ip == n {
                    // The insertion point is at the end of the old map, so insertion is straightforward
                    new_map.kv[..ip].copy_from_slice(self.keys());
                    new_map.kv[new_len..(new_len + ip)].copy_from_slice(self.values());
                    new_map.kv[ip] = key;
                    new_map.kv[new_len + ip] = value
                } else {
                    // The insertion point is in the middle, so we need to perform a series of split copies
                    //
                    // Slice up the old map into the prefix/suffix sections for both keys and values
                    let keys_prefix_old = &self.kv[..ip];
                    let keys_suffix_old = &self.kv[ip..n];
                    let values_prefix_old = &self.kv[n..(n + ip)];
                    let values_suffix_old = &self.kv[(n + ip)..];
                    // Copy the old values into the new map before/after the insertion point of the new key/value
                    new_map.kv[..ip].copy_from_slice(keys_prefix_old);
                    new_map.kv[(ip + 1)..new_len].copy_from_slice(keys_suffix_old);
                    new_map.kv[new_len..(new_len + ip)].copy_from_slice(values_prefix_old);
                    new_map.kv[(new_len + ip + 1)..].copy_from_slice(values_suffix_old);
                    new_map.kv[ip] = key;
                    new_map.kv[new_len + ip] = value;
                }
                Ok(new_map)
            }
        }
    }

    /// Updates `key` with `value` in `map`.
    ///
    /// If `key` does not exist in the map, an error is returned.
    pub fn update<K, V, A>(
        self: Gc<Self>,
        key: K,
        value: V,
        alloc: &A,
    ) -> Result<Gc<Self>, MapError>
    where
        K: Into<OpaqueTerm>,
        V: Into<OpaqueTerm>,
        A: ?Sized + Allocator,
    {
        self.do_update(key.into(), value.into(), alloc)
    }

    fn do_update<A>(
        self: Gc<Self>,
        key: OpaqueTerm,
        value: OpaqueTerm,
        alloc: &A,
    ) -> Result<Gc<Self>, MapError>
    where
        A: ?Sized + Allocator,
    {
        let n = self.size();
        if n == 0 {
            return Err(MapError::BadKey);
        }

        let index = find_key(&self, key).map_err(|_| MapError::BadKey)?;
        let capacity = self.capacity();
        let value_index = capacity + index;

        if self.kv[value_index] == value {
            // The value is unchanged, return a pointer to the original map
            Ok(self)
        } else if n == capacity {
            let mut new_map = Self::with_capacity_in(n, alloc)?;
            new_map.bitmap = u64::MAX << (64 - n);
            new_map.kv.copy_from_slice(&self.kv);
            new_map.kv[value_index] = value;
            Ok(new_map)
        } else {
            let mut new_map = Self::with_capacity_in(n, alloc)?;
            new_map.bitmap = u64::MAX << (64 - n);
            new_map.keys_mut().copy_from_slice(self.keys());
            new_map.values_mut().copy_from_slice(self.values());
            new_map.kv[n + index] = value;
            Ok(new_map)
        }
    }

    /// Takes the value associated with `key` out of `map`, returning both the value and a new map without `key`.
    ///
    /// If `key` is not found in `map`, an error is returned indicating the key is invalid.
    pub fn take<K, A>(self: Gc<Self>, key: K, alloc: &A) -> Result<(OpaqueTerm, Gc<Self>), MapError>
    where
        K: Into<OpaqueTerm>,
        A: ?Sized + Allocator,
    {
        self.do_take(key.into(), alloc)
    }

    fn do_take<A>(
        self: Gc<Self>,
        key: OpaqueTerm,
        alloc: &A,
    ) -> Result<(OpaqueTerm, Gc<Self>), MapError>
    where
        A: ?Sized + Allocator,
    {
        let n = self.size();
        if n == 0 {
            return Err(MapError::BadKey);
        }

        let index = find_key(&self, key).map_err(|_| MapError::BadKey)?;
        let capacity = self.capacity();
        let value_index = capacity + index;
        let new_capacity = n - 1;

        let value = self.kv[value_index];
        if new_capacity == 0 {
            return Ok((value, Self::new_in(alloc)?));
        }

        let mut new_map = Self::with_capacity_in(new_capacity, alloc)?;
        new_map.bitmap = u64::MAX << (64 - new_capacity);

        // If the index being removed occurs at the end, we can optimize the copy
        if index == n {
            let keys_old = &self.kv[..index];
            let values_old = &self.kv[capacity..value_index];
            let (keys, values) = new_map.kv.split_at_mut(new_capacity);
            keys.copy_from_slice(keys_old);
            values.copy_from_slice(values_old);
            return Ok((value, new_map));
        }

        // Likewise if the index occurs at the start
        if index == 0 {
            let keys_old = &self.kv[1..n];
            let values_old = &self.kv[(capacity + 1)..(capacity + n)];
            let (keys, values) = new_map.kv.split_at_mut(new_capacity);
            keys.copy_from_slice(keys_old);
            values.copy_from_slice(values_old);
            return Ok((value, new_map));
        }

        // Otherwise we have to do a more complicated set of split copies

        // Split the kv array excluding the key and value indices
        let keys_prefix_old = &self.kv[..index];
        let keys_suffix_old = &self.kv[(index + 1)..n];
        let values_prefix_old = &self.kv[capacity..value_index];
        let values_suffix_old = &self.kv[(value_index + 1)..(capacity + n)];
        // Then split the new kv array into equivalent sized chunks
        let (keys_prefix, rest) = new_map.kv.split_at_mut(index);
        debug_assert_eq!(keys_prefix.len(), keys_prefix_old.len());
        let (keys_suffix, rest) = rest.split_at_mut(new_capacity - index);
        debug_assert_eq!(keys_suffix.len(), keys_suffix_old.len());
        let (values_prefix, values_suffix) = rest.split_at_mut(index);
        debug_assert_eq!(values_prefix.len(), values_prefix_old.len());
        debug_assert_eq!(values_suffix.len(), values_suffix_old.len());
        keys_prefix.copy_from_slice(keys_prefix_old);
        keys_suffix.copy_from_slice(keys_suffix_old);
        values_prefix.copy_from_slice(values_prefix_old);
        values_suffix.copy_from_slice(values_suffix_old);
        Ok((value, new_map))
    }

    pub fn take_mut<K>(&mut self, key: K) -> Option<OpaqueTerm>
    where
        K: Into<OpaqueTerm>,
    {
        self.do_take_mut(key.into())
    }

    fn do_take_mut(&mut self, key: OpaqueTerm) -> Option<OpaqueTerm> {
        let size = self.size();
        let capacity = self.capacity();
        if size == 0 {
            return None;
        }

        let index = find_key(self, key).ok()?;

        // If the index to remove is last in the map, we can avoid shifting any elements
        if index == size - 1 {
            self.bitmap <<= 1;
            let value = self.kv[capacity + index];
            self.kv[index] = OpaqueTerm::NONE;
            self.kv[capacity + index] = OpaqueTerm::NONE;
            return Some(value);
        }

        // Otherwise we need to shift all of the elements after the removed index, left by 1 slot
        self.bitmap <<= 1;
        let value = self.kv[capacity + index];
        self.kv[index] = OpaqueTerm::NONE;
        self.kv[capacity + index] = OpaqueTerm::NONE;
        self.kv[index..size].rotate_left(1);
        self.kv[(capacity + index)..(capacity + size)].rotate_left(1);
        Some(value)
    }

    /// Merges two maps, producing a new map containing the union of all keys, preferring the values
    /// from `map2` when a conflicting key is present in both maps.
    pub fn merge<A: ?Sized + Allocator>(
        self: Gc<Self>,
        map2: &Gc<Self>,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        // In the unlikely case that the two maps are the same map, just clone one of them
        if unlikely(Gc::as_ptr(&self) == Gc::as_ptr(map2)) {
            return Ok(self);
        }

        let n1 = self.size();
        let n2 = map2.size();

        // If either map is empty, we can simply clone the other map as the result
        if n1 == 0 {
            return Ok(map2.clone());
        }
        if n2 == 0 {
            return Ok(self.clone());
        }

        // Allocate a temporary buffer on the stack while we determine the size of the final
        // map and its key ordering. The alternative is to
        let new_capacity = n1 + n2;
        let mut buffer = [OpaqueTerm::NONE; SMALL_MAP_LIMIT * 2];
        let new_kv = &mut buffer[..(new_capacity * 2)];

        let keys1 = self.keys();
        let keys2 = map2.keys();
        let values1 = self.values();
        let values2 = map2.values();
        let mut i = 0;
        let mut i1 = 0;
        let mut i2 = 0;
        let (new_keys, new_values) = unsafe { new_kv.split_at_mut_unchecked(new_capacity) };
        while i1 < n1 && i2 < n2 {
            use core::cmp::Ordering;

            let k1 = keys1[i1];
            let k2 = keys2[i2];
            match compare_keys(k1, k2) {
                Ordering::Equal => {
                    // Use right-hand side map's value, but advance both maps
                    new_keys[i] = k2;
                    new_values[i] = values2[i2];
                    i += 1;
                    i1 += 1;
                    i2 += 1;
                }
                Ordering::Less => {
                    new_keys[i] = k1;
                    new_values[i] = values1[i1];
                    i += 1;
                    i1 += 1;
                }
                Ordering::Greater => {
                    new_keys[i] = k2;
                    new_values[i] = values2[i2];
                    i += 1;
                    i2 += 1;
                }
            }
        }

        // Copy remaining
        if i1 < n1 {
            let rest = &keys1[i1..];
            let remaining = rest.len();
            let nk = &mut new_keys[i..(i + remaining)];
            nk.copy_from_slice(rest);

            let rest = &values1[i1..];
            let nv = &mut new_values[i..(i + remaining)];
            nv.copy_from_slice(rest);

            i += remaining;
        }
        if i2 < n2 {
            let rest = &keys2[i2..];
            let remaining = rest.len();
            let nk = &mut new_keys[i..(i + remaining)];
            nk.copy_from_slice(rest);

            let rest = &values2[i2..];
            let nv = &mut new_values[i..(i + remaining)];
            nv.copy_from_slice(rest);

            i += remaining;
        }

        // Allocate the correctly-sized map on the process heap and memcpy the
        // contents from our temporary map allocated on the stack
        let mut new_map = Self::with_capacity_in(i, alloc)?;
        new_map.bitmap = u64::MAX << (64 - i);
        new_map.keys_mut().copy_from_slice(&new_keys[..i]);
        new_map.values_mut().copy_from_slice(&new_values[..i]);
        Ok(new_map)
    }
}
impl fmt::Debug for SmallMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("#{")?;
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{:?} => {:?}", key, value)?;
        }
        f.write_str("}")
    }
}
impl fmt::Display for SmallMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("#{")?;
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} => {}", key, value)?;
        }
        f.write_str("}")
    }
}
impl Eq for SmallMap {}
impl PartialEq for SmallMap {
    fn eq(&self, other: &Self) -> bool {
        if self.size() == other.size() {
            self.iter().eq(other.iter())
        } else {
            false
        }
    }
}
impl PartialEq<Gc<SmallMap>> for SmallMap {
    fn eq(&self, other: &Gc<SmallMap>) -> bool {
        self.eq(other.deref())
    }
}
impl ExactEq for SmallMap {
    fn exact_eq(&self, other: &Self) -> bool {
        if self.size() == other.size() {
            self.iter().eq_by(other.iter(), |(ak, av), (bk, bv)| {
                if ak.exact_eq(&bk) {
                    av.exact_eq(&bv)
                } else {
                    false
                }
            })
        } else {
            false
        }
    }
}
impl Hash for SmallMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in self.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}
impl PartialOrd for SmallMap {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SmallMap {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        // Maps are ordered as follows:
        //
        // * First by size, with smaller maps being "less" than larger maps
        // * If the same size, then by keys in term order
        // * If the keys are the same, then by values in key order
        //
        // This corresponds to the lexicographical sorting of the underlying array
        match self.size().cmp(&other.size()) {
            Ordering::Equal => match self.keys().cmp(other.keys()) {
                Ordering::Equal => self.values().cmp(other.values()),
                other => other,
            },
            other => other,
        }
    }
}

pub struct SmallMapIter<'a> {
    map: &'a SmallMap,
    index: usize,
}
impl<'a> SmallMapIter<'a> {
    #[inline]
    fn new(map: &'a SmallMap) -> Self {
        Self { map, index: 0 }
    }
}
impl<'a> core::iter::FusedIterator for SmallMapIter<'a> {}
unsafe impl<'a> core::iter::TrustedLen for SmallMapIter<'a> {}
impl<'a> core::iter::ExactSizeIterator for SmallMapIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.map.size()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
impl<'a> Iterator for SmallMapIter<'a> {
    type Item = (Term, Term);

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.map.size();
        (size, Some(size))
    }

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.map.size();
        if self.index >= size {
            return None;
        }

        let capacity = self.map.capacity();
        unsafe {
            let key = *self.map.kv.get_unchecked(self.index);
            let value = *self.map.kv.get_unchecked(capacity + self.index);
            self.index += 1;
            Some((key.into(), value.into()))
        }
    }
}

/// SmallMap is ordered, so we can optimize the time it takes to find the insertion
/// point by performing a binary search rather than a linear one. If found, `Ok` is
/// returned with the index of the matching key. If not, `Err` tells us at which index
/// to insert to maintain the sort order.
#[inline]
fn find_key(map: &SmallMap, key: OpaqueTerm) -> Result<usize, usize> {
    map.keys()
        .binary_search_by(|probe| compare_keys(*probe, key))
}

#[inline]
fn compare_keys(k1: OpaqueTerm, k2: OpaqueTerm) -> core::cmp::Ordering {
    use core::cmp::Ordering;

    match k1.cmp(&k2) {
        Ordering::Equal => {
            // We have to perform an exact comparison to "confirm" equality
            // of terms here, as term comparison for ordering does not use
            // strict equality as an optimization.
            if k1.exact_eq(&k2) {
                Ordering::Equal
            } else {
                // This is arbitrary, but suitable for our purposes
                Ordering::Less
            }
        }
        ord => ord,
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapFlags(usize);
impl MapFlags {
    const TAG_MASK: usize = 0b11;
    const VAL_MASK: usize = !Self::TAG_MASK;

    const FLATMAP: usize = 0b00;
    const NODE: usize = 0b01;
    const HEAD_ARRAY_NODE: usize = 0b10;
    const HEAD_BMAP_NODE: usize = 0b11;

    pub fn is_flatmap(&self) -> bool {
        self.0 & Self::TAG_MASK == Self::FLATMAP
    }

    pub fn is_head(&self) -> bool {
        let tag = self.0 & Self::TAG_MASK;
        tag == Self::HEAD_ARRAY_NODE || tag == Self::HEAD_BMAP_NODE
    }

    pub fn is_array_head(&self) -> bool {
        self.0 & Self::TAG_MASK == Self::HEAD_ARRAY_NODE
    }

    pub fn is_node(&self) -> bool {
        self.0 & Self::TAG_MASK == Self::NODE
    }

    pub fn capacity(&self) -> usize {
        assert!(self.is_flatmap());
        (self.0 & Self::VAL_MASK) >> 2
    }

    pub fn bitmap(&self) -> usize {
        assert!(!self.is_flatmap());
        (self.0 & Self::VAL_MASK) >> 2
    }
}
impl Metadata<SmallMap> for MapFlags {
    fn metadata(&self) -> <SmallMap as Pointee>::Metadata {
        self.capacity() * 2
    }
    fn pack(self) -> usize {
        self.0
    }
    unsafe fn unpack(raw: usize) -> Self {
        Self(raw)
    }
}

impl Boxable for SmallMap {
    type Metadata = MapFlags;

    const TAG: Tag = Tag::Map;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains((self as *const Self).cast()) {
            return Layout::new::<()>();
        }
        let mut builder = LayoutBuilder::new();
        for k in self.keys().iter().copied() {
            if !k.is_gcbox() {
                continue;
            }
            let term: Term = k.into();
            builder.extend(&term);
        }
        for v in self.values().iter().copied() {
            if !v.is_gcbox() {
                continue;
            }
            let term: Term = v.into();
            builder.extend(&term);
        }
        unsafe {
            let placeholder: *const SmallMap = ptr::from_raw_parts(ptr::null(), self.size());
            builder += Layout::for_value_raw(placeholder);
        }
        builder.finish()
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let size = self.size();
            let capacity = self.capacity();
            let mut map = Self::with_capacity_in(size, heap).unwrap();
            for (i, k) in self.keys().iter().copied().enumerate() {
                if !k.is_gcbox() {
                    k.maybe_increment_refcount();
                    unsafe {
                        *map.kv.get_unchecked_mut(i) = k;
                    }
                    continue;
                }
                let k: Term = k.into();
                unsafe {
                    *map.kv.get_unchecked_mut(i) = k.unsafe_clone_to_heap(heap).into();
                }
            }
            for (i, v) in self.values().iter().copied().enumerate() {
                if !v.is_gcbox() {
                    v.maybe_increment_refcount();
                    unsafe {
                        *map.kv.get_unchecked_mut(capacity + i) = v;
                    }
                    continue;
                }
                let v: Term = v.into();
                unsafe {
                    *map.kv.get_unchecked_mut(capacity + i) = v.unsafe_clone_to_heap(heap).into();
                }
            }
            map
        }
    }
}
impl SmallMap {
    pub unsafe fn unsafe_move_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        use crate::term::Cons;

        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let size = self.size();
            let capacity = self.capacity();
            let mut map = Self::with_capacity_in(size, heap).unwrap();
            for (i, k) in self.keys().iter().copied().enumerate() {
                if k.is_rc() {
                    *map.kv.get_unchecked_mut(i) = k;
                } else if k.is_nonempty_list() {
                    let mut cons = Gc::from_raw(k.as_ptr() as *mut Cons);
                    let moved = cons.unsafe_move_to_heap(heap);
                    *map.kv.get_unchecked_mut(i) = moved.into();
                } else if k.is_gcbox() || k.is_tuple() {
                    let term: Term = k.into();
                    let moved = term.unsafe_move_to_heap(heap);
                    *map.kv.get_unchecked_mut(i) = moved.into();
                } else {
                    *map.kv.get_unchecked_mut(i) = k;
                }
            }
            for (i, v) in self.values().iter().copied().enumerate() {
                if v.is_rc() {
                    *map.kv.get_unchecked_mut(capacity + i) = v;
                } else if v.is_nonempty_list() {
                    let mut cons = Gc::from_raw(v.as_ptr() as *mut Cons);
                    let moved = cons.unsafe_move_to_heap(heap);
                    *map.kv.get_unchecked_mut(capacity + i) = moved.into();
                } else if v.is_gcbox() || v.is_tuple() {
                    let term: Term = v.into();
                    let moved = term.unsafe_move_to_heap(heap);
                    *map.kv.get_unchecked_mut(capacity + i) = moved.into();
                } else {
                    *map.kv.get_unchecked_mut(capacity + i) = v;
                }
            }
            map
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use firefly_alloc::heap::FixedSizeHeap;

    use crate::term::*;

    #[test]
    fn smallmap_integration_test() {
        let mut map = SmallMap::with_capacity(10);
        assert_eq!(map.size(), 0);
        assert_eq!(map.capacity(), 10);
        assert_eq!(map.kv.len(), 20);
        assert_eq!(map.bitmap, 0);
        assert_eq!(map.header.tag(), Tag::Map);

        // Insert a key
        map.put_mut(atoms::True, Term::Int(101));
        assert_eq!(map.size(), 1);
        assert_eq!(map.capacity(), 10);
        assert_eq!(map.kv.len(), 20);
        assert_eq!(map.bitmap, 1 << 63);

        // Should be able to fetch that key
        assert_eq!(map.get(atoms::True).map(|t| t.into()), Some(Term::Int(101)));

        // Should be able to take that key out
        assert_eq!(
            map.take_mut(atoms::True).map(|t| t.into()),
            Some(Term::Int(101))
        );
        assert_eq!(map.size(), 0);
        assert_eq!(map.capacity(), 10);
        assert_eq!(map.kv.len(), 20);
        assert_eq!(map.bitmap, 0);

        // Shouldn't be able to get the key after removal
        assert_eq!(map.get(atoms::True), None);
    }

    #[test]
    fn smallmap_put_test() {
        let heap = FixedSizeHeap::<1024>::default();

        let map = SmallMap::new_in(&heap).unwrap();
        assert!(map.is_empty());

        let map2 = map.put(Term::Int(3), Term::Bool(true), &heap).unwrap();
        assert!(!map2.is_empty());
        assert_eq!(map2.size(), 1);
        assert_eq!(map2.capacity(), 1);
        assert_eq!(map2.kv.len(), 2);
        assert_eq!(map2.bitmap, 1 << 63);

        // Insert an item at the beginning
        let map3 = map2.put(Term::Int(1), Term::Bool(true), &heap).unwrap();
        assert!(!map3.is_empty());
        assert_eq!(map3.size(), 2);
        assert_eq!(map3.capacity(), 2);
        assert_eq!(map3.kv.len(), 4);
        assert_eq!(map3.bitmap, (1 << 63) | (1 << 62));

        // Insert an item at the end
        let map4 = map3.put(Term::Int(4), Term::Bool(true), &heap).unwrap();
        assert!(!map4.is_empty());
        assert_eq!(map4.size(), 3);
        assert_eq!(map4.capacity(), 3);
        assert_eq!(map4.kv.len(), 6);
        assert_eq!(map4.bitmap, (1 << 63) | (1 << 62) | (1 << 61));

        // Insert an item in the middle
        let map5 = map4.put(Term::Int(2), Term::Bool(true), &heap).unwrap();
        assert!(!map5.is_empty());
        assert_eq!(map5.size(), 4);
        assert_eq!(map5.capacity(), 4);
        assert_eq!(map5.kv.len(), 8);
        assert_eq!(map5.bitmap, (1 << 63) | (1 << 62) | (1 << 61) | (1 << 60));

        // Update an item
        let map6 = map5.put(Term::Int(1), Term::Bool(false), &heap).unwrap();
        // Verify that the update did not affect the original map
        assert!(!map5.is_empty());
        assert_eq!(map5.size(), 4);
        assert_eq!(map5.capacity(), 4);
        assert_eq!(map5.kv.len(), 8);
        assert_eq!(map5.bitmap, (1 << 63) | (1 << 62) | (1 << 61) | (1 << 60));

        // Verify that the update did not change the size of the map
        assert!(!map6.is_empty());
        assert_eq!(map6.size(), 4);
        assert_eq!(map6.capacity(), 4);
        assert_eq!(map6.kv.len(), 8);
        assert_eq!(map6.bitmap, (1 << 63) | (1 << 62) | (1 << 61) | (1 << 60));

        // Verify that the keys are ordered properly
        let expected: &[OpaqueTerm] = &[
            Term::Int(1).into(),
            Term::Int(2).into(),
            Term::Int(3).into(),
            Term::Int(4).into(),
        ];
        assert_eq!(map6.keys(), expected);

        // Verify that the values of the pre-update map are as expected
        let expected: &[OpaqueTerm] = &[
            atoms::True.into(),
            atoms::True.into(),
            atoms::True.into(),
            atoms::True.into(),
        ];
        assert_eq!(map5.values(), expected);

        // Verify that the values of the post-update map are as expected
        let expected: &[OpaqueTerm] = &[
            atoms::False.into(),
            atoms::True.into(),
            atoms::True.into(),
            atoms::True.into(),
        ];
        assert_eq!(map6.values(), expected);
    }

    #[test]
    fn smallmap_take_test() {
        let heap = FixedSizeHeap::<256>::default();
        let mut map = SmallMap::with_capacity_in(3, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        map.put_mut(Term::Int(2), Term::Bool(false));
        map.put_mut(Term::Int(3), Term::Atom(atoms::Undefined));

        // Try to take a non-existent value
        assert_eq!(map.take(Term::Int(10), &heap), Err(MapError::BadKey));

        // Take a value from the start
        let (value, map2) = map.take(Term::Int(1), &heap).unwrap();

        // Verify the value we took out
        assert_eq!(Into::<Term>::into(value), Term::Bool(true));
        // Verify the new map no longer has that element
        assert!(!map2.is_empty());
        assert_eq!(map2.size(), 2);
        assert_eq!(map2.capacity(), 2);
        assert_eq!(map2.bitmap, (1 << 63) | (1 << 62));

        // Verify the original map is unmodified
        assert!(!map.is_empty());
        assert_eq!(map.size(), 3);
        assert_eq!(map.capacity(), 3);
        assert_eq!(map.bitmap, 0b111 << 61);

        // Take a value from the end
        let (value, map3) = map.take(Term::Int(3), &heap).unwrap();
        assert_eq!(Into::<Term>::into(value), Term::Atom(atoms::Undefined));
        assert!(!map3.is_empty());
        assert_eq!(map3.size(), 2);
        assert_eq!(map3.capacity(), 2);
        assert_eq!(map3.bitmap, (1 << 63) | (1 << 62));

        // Take a value from the middle
        let (value, map4) = map.take(Term::Int(2), &heap).unwrap();
        assert_eq!(Into::<Term>::into(value), Term::Bool(false));
        assert!(!map4.is_empty());
        assert_eq!(map4.size(), 2);
        assert_eq!(map4.capacity(), 2);
        assert_eq!(map4.bitmap, (1 << 63) | (1 << 62));
    }

    #[test]
    fn smallmap_update_test() {
        let heap = FixedSizeHeap::<256>::default();

        let mut map = SmallMap::with_capacity_in(3, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        map.put_mut(Term::Int(2), Term::Bool(false));
        map.put_mut(Term::Int(3), Term::Atom(atoms::Undefined));

        // Try to update non-existent key
        assert_eq!(
            map.update(Term::Int(10), Term::Atom(atoms::Ok), &heap),
            Err(MapError::BadKey)
        );

        // Update element at start
        let map2 = map
            .update(Term::Int(1), Term::Atom(atoms::Ok), &heap)
            .unwrap();
        assert_eq!(map2.is_empty(), map.is_empty());
        assert_eq!(map2.size(), map.size());
        assert_eq!(map2.bitmap, map.bitmap);

        // Update element at end
        let map3 = map2
            .update(Term::Int(3), Term::Atom(atoms::Error), &heap)
            .unwrap();
        assert_eq!(map3.is_empty(), map2.is_empty());
        assert_eq!(map3.size(), map2.size());
        assert_eq!(map3.bitmap, map2.bitmap);

        // Update element in middle
        let map4 = map3
            .update(Term::Int(2), Term::Atom(atoms::Warning), &heap)
            .unwrap();
        assert_eq!(map4.is_empty(), map3.is_empty());
        assert_eq!(map4.size(), map3.size());
        assert_eq!(map4.bitmap, map3.bitmap);

        let expected: &[OpaqueTerm] = &[
            Term::Int(1).into(),
            Term::Int(2).into(),
            Term::Int(3).into(),
        ];
        assert_eq!(map4.keys(), expected);

        let expected: &[OpaqueTerm] = &[
            Term::Atom(atoms::Ok).into(),
            Term::Atom(atoms::Warning).into(),
            Term::Atom(atoms::Error).into(),
        ];
        assert_eq!(map4.values(), expected);
    }

    #[test]
    fn smallmap_merge_test() {
        let heap = FixedSizeHeap::<384>::default();

        let empty = SmallMap::new_in(&heap).unwrap();

        let mut map = SmallMap::with_capacity_in(2, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        map.put_mut(Term::Int(2), Term::Bool(false));

        // Merging a non-empty map into an empty one should return the non-empty map
        let merged = empty.merge(&map, &heap).unwrap();
        assert_eq!(Gc::as_ptr(&map), Gc::as_ptr(&merged));

        // Merging an empty map into a non-empty one should return the non-empty unmodified
        let merged = map.merge(&empty, &heap).unwrap();
        assert_eq!(Gc::as_ptr(&map), Gc::as_ptr(&merged));

        // Create another map with some elements in common with `map`
        //
        // NOTE: We explicitly leave unused capacity in this map to test the merge behavior
        let mut map2 = SmallMap::with_capacity_in(4, &heap).unwrap();
        map2.put_mut(Term::Int(2), Term::Bool(true));
        map2.put_mut(Term::Int(4), Term::Bool(true));

        // Merging map2 into map should prefer map2's keys
        let merged = map.merge(&map2, &heap).unwrap();
        assert_ne!(Gc::as_ptr(&map), Gc::as_ptr(&merged));
        assert!(!merged.is_empty());
        assert_eq!(merged.size(), 3);
        assert_eq!(merged.capacity(), 3);

        let expected: &[OpaqueTerm] = &[
            Term::Int(1).into(),
            Term::Int(2).into(),
            Term::Int(4).into(),
        ];
        assert_eq!(merged.keys(), expected);

        let expected: &[OpaqueTerm] = &[
            Term::Bool(true).into(),
            Term::Bool(true).into(),
            Term::Bool(true).into(),
        ];
        assert_eq!(merged.values(), expected);

        // Merging map into map2 should prefer map's keys
        let merged = map2.merge(&map, &heap).unwrap();
        assert_ne!(Gc::as_ptr(&map2), Gc::as_ptr(&merged));
        assert!(!merged.is_empty());
        assert_eq!(merged.size(), 3);
        assert_eq!(merged.capacity(), 3);

        let expected: &[OpaqueTerm] = &[
            Term::Int(1).into(),
            Term::Int(2).into(),
            Term::Int(4).into(),
        ];
        assert_eq!(merged.keys(), expected);

        let expected: &[OpaqueTerm] = &[
            Term::Bool(true).into(),
            Term::Bool(false).into(),
            Term::Bool(true).into(),
        ];
        assert_eq!(merged.values(), expected);
    }

    #[test]
    fn smallmap_compare_keys_test() {
        let heap = FixedSizeHeap::<256>::default();

        // Equivalent numeric values should not compare equal if they are not the same type
        assert_ne!(
            super::compare_keys(Term::Int(101).into(), 101.0.into()),
            Ordering::Equal
        );
        // Conversely, they should if they are the same type
        assert_eq!(
            super::compare_keys(Term::Int(101).into(), Term::Int(101).into()),
            Ordering::Equal
        );

        // Comparing binaries for equality should not be type sensitive, as long as data is equivalent
        let bin = BinaryData::from_small_str("foo", &heap).unwrap();
        let slice = unsafe { BitSlice::new(bin.into(), bin.as_bytes(), 0, bin.bit_size()) };
        let bin1 = Term::HeapBinary(bin);
        let bin2 = Term::RefBinary(Gc::new_in(slice, &heap).unwrap());
        assert_eq!(
            super::compare_keys(bin1.into(), bin2.into()),
            Ordering::Equal
        );
    }
}
