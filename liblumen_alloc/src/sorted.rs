///! This module provides a reusable sorted key adapter for use
///! with intrusive collections, namely `RBTree`.
///!
///! It provides additional auxilary types and traits which are
///! either used by `SortedKeyAdapter` or by types which need to
///! be stored in a collection managed by that adapter.
use core::marker::PhantomData;
use core::cmp::Ordering;

use intrusive_collections::{UnsafeRef, Adapter, KeyAdapter};
use intrusive_collections::{RBTreeLink, LinkedListLink};

/// A simple marker trait for intrusive collection links
pub trait Link {}
impl Link for RBTreeLink {}
impl Link for LinkedListLink {}


/// This trait is used to make the sorted key adapter more general
/// by delegating some of the work to the type being stored in the
/// intrusive collection
pub trait Sortable {
    type Link: Link;

    /// Get a pointer to the value from the given link pointer.
    ///
    /// The sort order used for the container is given and is used
    /// to differentiate between links if many are present
    fn get_value(link: *const Self::Link, order: SortOrder) -> *const Self;

    /// Get a link pointer from the given value.
    ///
    /// The sort order used for the container is given and is used
    /// to differentiate between links if many are present
    fn get_link(value: *const Self, order: SortOrder) -> *const Self::Link;

    /// Get the sort key to use for this value.
    ///
    /// The sort order provided may be used to determine what values
    /// to place in the key, and should be used as the order given
    /// to `SortKey`
    fn sort_key(&self, order: SortOrder) -> SortKey;
}

/// This enum provides a means by which the sort order used
/// with the adapter can be dynamically changed.
///
/// It also facilitates expressing complex orderings, such
/// as "order by size, then address order".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// just address order
    AddressOrder,
    /// first size (largest to smallest), then address order
    SizeAddressOrder,
}

/// This struct is used as the sorting key when determing relative
/// order between elements in an ordered collection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortKey(SortOrder, usize, usize);
impl SortKey {
    pub const fn new(order: SortOrder, size: usize, addr: usize) -> Self {
        Self(order, size, addr)
    }
}
impl Ord for SortKey {
    #[inline]
    fn cmp(&self, other: &SortKey) -> Ordering {
        match self.0 {
            SortOrder::AddressOrder =>
                self.2.cmp(&other.2),
            SortOrder::SizeAddressOrder =>
                match self.1.cmp(&other.1) {
                    Ordering::Equal => self.2.cmp(&other.2),
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                },
        }
    }
}
impl PartialOrd for SortKey {
    #[inline]
    fn partial_cmp(&self, other: &SortKey) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


/// This struct is the primary point of this module; an adapter
/// for sorted intrusive collections that allows reacting to
/// the desired order dynamically at runtime, rather than statically.
///
/// Currently, this is designed for use with `RBTree`, but could
/// be made to support other collections if desired.
pub struct SortedKeyAdapter<T>
where
    T: Sortable,
{
    order: SortOrder,
    _phantom: PhantomData<UnsafeRef<T>>
}

impl<T> SortedKeyAdapter<T>
where
    T: Sortable,
{
    pub fn new(order: SortOrder) -> Self {
        Self { order, _phantom: PhantomData }
    }
}

unsafe impl<T> Send for SortedKeyAdapter<T> where T: Sortable {}
unsafe impl<T> Sync for SortedKeyAdapter<T> where T: Sortable {}

impl<T> Copy for SortedKeyAdapter<T>
where
    T: Sortable + Copy,
{}

impl<T> Clone for SortedKeyAdapter<T>
where
    T: Sortable + Clone,
{
    fn clone(&self) -> Self {
        let order = self.order;
        SortedKeyAdapter { order, _phantom: PhantomData }
    }
}

#[allow(dead_code, unsafe_code)]
unsafe impl<T> Adapter for SortedKeyAdapter<T>
where
    T: Sortable<Link = RBTreeLink>,
{
    // Currently the only sorted collection is RBTree
    type Link = <T as Sortable>::Link;
    // The value type actually referenced by nodes
    type Value = T;
    // The type of pointer stored in the tree
    type Pointer = UnsafeRef<T>;

    #[inline]
    unsafe fn get_value(&self, link: *const Self::Link) -> *const Self::Value {
        <T as Sortable>::get_value(link, self.order)
    }

    #[inline]
    unsafe fn get_link(&self, value: *const Self::Value) -> *const Self::Link {
        <T as Sortable>::get_link(value, self.order)
    }
}

impl<'a, T> KeyAdapter<'a> for SortedKeyAdapter<T>
where
    T: Sortable<Link = RBTreeLink>,
{
    type Key = SortKey;

    fn get_key(&self, value: &'a Self::Value) -> Self::Key {
        value.sort_key(self.order)
    }
}
