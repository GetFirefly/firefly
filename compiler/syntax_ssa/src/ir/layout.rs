use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

use cranelift_entity::EntityRef;
use firefly_arena::TypedArena;
use intrusive_collections::linked_list::{Cursor, CursorMut, LinkedList};
use intrusive_collections::{intrusive_adapter, LinkedListLink, UnsafeRef};

/// This struct holds the data for each node in an ArenaMap/OrderedArenaMap
#[derive(Clone)]
pub struct LayoutNode<K: EntityRef, V: Clone> {
    pub link: LinkedListLink,
    key: K,
    value: V,
}
impl<K: EntityRef, V: Clone> LayoutNode<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self {
            link: LinkedListLink::default(),
            key,
            value,
        }
    }

    #[inline(always)]
    pub fn key(&self) -> K {
        self.key
    }

    #[inline(always)]
    pub fn value(&self) -> &V {
        &self.value
    }

    #[inline(always)]
    pub fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

intrusive_adapter!(pub LayoutAdapter<K, V> = UnsafeRef<LayoutNode<K, V>>: LayoutNode<K, V> { link: LinkedListLink } where K: EntityRef, V: Clone);

/// ArenaMap provides similar functionality to other kinds of maps:
///
/// # Pros
///
/// * Once allocated, values stored in the map have a stable location, this can be useful for when
///   you
/// expect to store elements of the map in an intrusive collection.
/// * Keys can be more efficiently sized, i.e. rather than pointers/usize keys, you can choose
///   arbitrarily
/// small bitwidths, as long as there is sufficient keyspace for your use case.
/// * Attempt to keep data in the map as contiguous in memory as possible. This is again useful for
///   when
/// the data is also linked into an intrusive collection, like a linked list, where traversing the
/// list will end up visiting many of the nodes in the map. If each node was its own Box, this would
/// cause thrashing of the cache - ArenaMap sidesteps this by allocating values in chunks of memory
/// that are friendlier to the cache.
///
/// # Cons
///
/// * Memory allocated for data stored in the map is not released until the map is dropped. This is
/// a tradeoff made to ensure that the data has a stable location in memory, but the flip side of
/// that is increased memory usage for maps that stick around for a long time. In our case, these
/// maps are relatively short-lived, so it isn't a problem in practice.
/// * It doesn't provide as rich of an API as HashMap and friends
pub struct ArenaMap<K: EntityRef, V: Clone> {
    keys: Vec<Option<NonNull<V>>>,
    arena: TypedArena<V>,
    _marker: core::marker::PhantomData<K>,
}
impl<K: EntityRef, V: Clone> Drop for ArenaMap<K, V> {
    fn drop(&mut self) {
        self.keys.clear()
    }
}
impl<K: EntityRef, V: Clone> Clone for ArenaMap<K, V> {
    fn clone(&self) -> Self {
        let mut cloned = Self::new();
        for opt in self.keys.iter() {
            match opt {
                None => cloned.keys.push(None),
                Some(nn) => {
                    let value = unsafe { nn.as_ref() };
                    cloned.push(value.clone());
                }
            }
        }
        cloned
    }
}
impl<K: EntityRef, V: Clone> Default for ArenaMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<K: EntityRef, V: Clone> ArenaMap<K, V> {
    /// Creates a new, empty ArenaMap
    pub fn new() -> Self {
        Self {
            arena: TypedArena::default(),
            keys: vec![],
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the total number of actively linked items in the map
    pub fn len(&self) -> usize {
        self.keys.iter().filter(|item| item.is_some()).count()
    }

    /// Returns true if this map contains `key`
    pub fn contains(&self, key: K) -> bool {
        self.keys
            .get(key.index())
            .map(|item| item.is_some())
            .unwrap_or(false)
    }

    /// Adds a new entry to the map, returning the key it is associated to
    pub fn push(&mut self, value: V) -> K {
        let key = self.alloc_key();
        self.alloc_node(key, value);
        key
    }

    /// Used in conjunction with `alloc_key` to associate data with the allocated key
    pub fn append(&mut self, key: K, value: V) {
        self.alloc_node(key, value);
    }

    /// Returns a reference to the value associated with the given key
    pub fn get(&self, key: K) -> Option<&V> {
        self.keys
            .get(key.index())
            .and_then(|item| item.map(|nn| unsafe { nn.as_ref() }))
    }

    /// Returns a mutable reference to the value associated with the given key
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.keys
            .get_mut(key.index())
            .and_then(|item| item.map(|mut nn| unsafe { nn.as_mut() }))
    }

    /// Takes the value that was stored at the given key
    pub fn take(&mut self, key: K) -> Option<NonNull<V>> {
        self.keys[key.index()].take()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Option<NonNull<V>>> + 'a {
        self.keys.iter().copied()
    }

    /// Removes the value associated with the given key
    ///
    /// NOTE: This function will panic if the key is invalid/unbound
    pub fn remove(&mut self, key: K) {
        self.keys[key.index()].take();
    }

    /// Clears all of the data from the map
    pub fn clear(&mut self) {
        self.keys.clear();
        self.arena.clear();
    }

    pub fn alloc_key(&mut self) -> K {
        let id = self.keys.len();
        let key = K::new(id);
        self.keys.push(None);
        key
    }

    fn alloc_node(&mut self, key: K, value: V) -> NonNull<V> {
        let value = self.arena.alloc(value);
        let nn = unsafe { NonNull::new_unchecked(value) };
        self.keys[key.index()].replace(nn);
        nn
    }
}
impl<K: EntityRef, V: Clone> Index<K> for ArenaMap<K, V> {
    type Output = V;

    #[inline]
    fn index(&self, index: K) -> &Self::Output {
        self.get(index).unwrap()
    }
}
impl<K: EntityRef, V: Clone> IndexMut<K> for ArenaMap<K, V> {
    #[inline]
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

/// OrderedArenaMap is an extension of ArenaMap that provides for arbitrary ordering of keys/values
///
/// This is done using an intrusive linked list alongside an ArenaMap. The list is used to link one
/// key/value pair to the next, so any ordering you wish to implement is possible. This is
/// particularly useful for layout of blocks in a function, or instructions within blocks, as you
/// can precisely position them relative to other blocks/instructions.
///
/// Because the linked list is intrusive, it is virtually free in terms of space, but comes with the
/// standard overhead for traversals. That said, there are a couple of niceties that give it good
/// overall performance:
///
/// * It is a doubly-linked list, so you can traverse equally efficiently front-to-back or
///   back-to-front,
/// * It has O(1) indexing; given a key, we can directly obtain a reference to a node, and with
///   that,
/// obtain a cursor over the list starting at that node.
pub struct OrderedArenaMap<K: EntityRef, V: Clone> {
    list: LinkedList<LayoutAdapter<K, V>>,
    map: ArenaMap<K, LayoutNode<K, V>>,
}
impl<K: EntityRef, V: Clone> Drop for OrderedArenaMap<K, V> {
    fn drop(&mut self) {
        self.list.fast_clear();
    }
}
impl<K: EntityRef, V: Clone> Clone for OrderedArenaMap<K, V> {
    fn clone(&self) -> Self {
        let mut cloned = Self::new();
        for opt in self.map.iter() {
            match opt {
                None => {
                    cloned.map.alloc_key();
                }
                Some(nn) => {
                    let value = unsafe { nn.as_ref() }.value();
                    cloned.push(value.clone());
                }
            }
        }
        cloned
    }
}
impl<K: EntityRef, V: Clone> Default for OrderedArenaMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<K: EntityRef, V: Clone> OrderedArenaMap<K, V> {
    pub fn new() -> Self {
        Self {
            map: ArenaMap::new(),
            list: LinkedList::new(LayoutAdapter::new()),
        }
    }

    pub fn clear(&mut self) {
        self.list.fast_clear();
        self.map.clear();
    }

    /// Returns the total number of actively linked items in the map
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if this map contains the given key and its value has been linked
    #[inline]
    pub fn contains(&self, key: K) -> bool {
        self.map.contains(key)
    }

    /// Returns a reference to the value associated with the given key, if present and linked
    pub fn get(&self, key: K) -> Option<&V> {
        self.map.get(key).map(|data| data.value())
    }

    /// Returns a mutable reference to the value associated with the given key, if present and
    /// linked
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.map.get_mut(key).map(|data| data.value_mut())
    }

    /// Allocates a key, but does not link the data
    #[inline]
    pub fn create(&mut self) -> K {
        self.map.alloc_key()
    }

    /// Used with `create` when ready to associate data with the allocated key, linking it in to the
    /// end of the list
    pub fn append(&mut self, key: K, value: V) {
        debug_assert!(!self.contains(key));
        let data = self.alloc_node(key, value);
        self.list.push_back(data);
    }

    /// Like `append`, but inserts the node before `before` in the list
    ///
    /// NOTE: This function will panic if `before` is not present in the list
    pub fn insert_before(&mut self, key: K, before: K, value: V) {
        let value_opt = self.get_mut(key);
        debug_assert!(value_opt.is_none());
        let data = self.alloc_node(key, value);
        let mut cursor = self.cursor_mut_at(before);
        cursor.insert_before(data);
    }

    /// Like `append`, but inserts the node after `after` in the list
    ///
    /// NOTE: This function will panic if `after` is not present in the list
    pub fn insert_after(&mut self, key: K, after: K, value: V) {
        let value_opt = self.get_mut(key);
        debug_assert!(value_opt.is_none());
        let data = self.alloc_node(key, value);
        let mut cursor = self.cursor_mut_at(after);
        cursor.insert_after(data);
    }

    /// Allocates a key and links data in the same operation
    pub fn push(&mut self, value: V) -> K {
        let key = self.alloc_key();
        self.append(key, value);
        key
    }

    /// Unlinks the value associated with the given key from this map
    ///
    /// NOTE: Removal does not result in deallocation of the underlying data, this
    /// happens when the map is dropped. To perform early garbage collection, you can
    /// clone the map, and drop the original.
    pub fn remove(&mut self, key: K) {
        if let Some(nn) = self.map.take(key) {
            let mut cursor = unsafe { self.list.cursor_mut_from_ptr(nn.as_ptr()) };
            cursor.remove();
        }
    }

    /// Returns the first node in the map
    pub fn first(&self) -> Option<&LayoutNode<K, V>> {
        self.list.front().get()
    }

    /// Returns the last node in the map
    pub fn last(&self) -> Option<&LayoutNode<K, V>> {
        self.list.back().get()
    }

    /// Returns a cursor which can be used to traverse the map in order (front to back)
    pub fn cursor(&self) -> Cursor<'_, LayoutAdapter<K, V>> {
        self.list.front()
    }

    /// Returns a cursor which can be used to traverse the map mutably, in order (front to back)
    pub fn cursor_mut(&mut self) -> CursorMut<'_, LayoutAdapter<K, V>> {
        self.list.front_mut()
    }

    /// Returns a cursor which can be used to traverse the map in order (front to back), starting
    /// at the key given.
    pub fn cursor_at(&self, key: K) -> Cursor<'_, LayoutAdapter<K, V>> {
        let ptr = &self.map[key] as *const LayoutNode<K, V>;
        unsafe { self.list.cursor_from_ptr(ptr) }
    }

    /// Returns a cursor which can be used to traverse the map mutably, in order (front to back),
    /// starting at the key given.
    pub fn cursor_mut_at(&mut self, key: K) -> CursorMut<'_, LayoutAdapter<K, V>> {
        let ptr = &self.map[key] as *const LayoutNode<K, V>;
        unsafe { self.list.cursor_mut_from_ptr(ptr) }
    }

    /// Returns an iterator over the key/value pairs in the map, in order (front to back)
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (K, &'a V)> {
        self.list.iter().map(|item| (item.key(), item.value()))
    }

    /// Returns an iterator over the keys in the map, in order (front to back)
    pub fn keys<'a>(&'a self) -> impl Iterator<Item = K> + 'a {
        self.list.iter().map(|item| item.key())
    }

    /// Returns an iterator over the values in the map, in order (front to back)
    pub fn values<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        self.list.iter().map(|item| item.value())
    }

    #[inline]
    fn alloc_key(&mut self) -> K {
        self.map.alloc_key()
    }

    fn alloc_node(&mut self, key: K, value: V) -> UnsafeRef<LayoutNode<K, V>> {
        let nn = self.map.alloc_node(key, LayoutNode::new(key, value));
        unsafe { UnsafeRef::from_raw(nn.as_ptr()) }
    }
}
impl<K: EntityRef, V: Clone> Index<K> for OrderedArenaMap<K, V> {
    type Output = V;

    #[inline]
    fn index(&self, index: K) -> &Self::Output {
        self.get(index).unwrap()
    }
}
impl<K: EntityRef, V: Clone> IndexMut<K> for OrderedArenaMap<K, V> {
    #[inline]
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}
