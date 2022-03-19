use core::cmp::Ordering;
///! This module provides a reusable sorted key adapter for use
///! with intrusive collections, namely `RBTree`.
///!
///! It provides additional auxilary types and traits which are
///! either used by `SortedKeyAdapter` or by types which need to
///! be stored in a collection managed by that adapter.
use core::marker::PhantomData;

use intrusive_collections::{Adapter, KeyAdapter, UnsafeRef};
use intrusive_collections::{DefaultPointerOps, LinkOps, LinkedListLink, PointerOps, RBTreeLink};

pub trait Link: Default + Clone {
    type LinkOps: LinkOps + Default + Clone;
}
impl Link for RBTreeLink {
    type LinkOps = intrusive_collections::rbtree::LinkOps;
}
impl Link for LinkedListLink {
    type LinkOps = intrusive_collections::linked_list::LinkOps;
}

/// This trait is used to make the sorted key adapter more general
/// by delegating some of the work to the type being stored in the
/// intrusive collection
pub trait Sortable {
    type Link: Link;

    /// Get a pointer to the value from the given link pointer.
    ///
    /// The sort order used for the container is given and is used
    /// to differentiate between links if many are present
    fn get_value(
        link: <<Self::Link as Link>::LinkOps as LinkOps>::LinkPtr,
        order: SortOrder,
    ) -> *const Self;

    /// Get a link pointer from the given value.
    ///
    /// The sort order used for the container is given and is used
    /// to differentiate between links if many are present
    fn get_link(
        value: *const Self,
        order: SortOrder,
    ) -> <<Self::Link as Link>::LinkOps as LinkOps>::LinkPtr;

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
            SortOrder::AddressOrder => self.2.cmp(&other.2),
            SortOrder::SizeAddressOrder => match self.1.cmp(&other.1) {
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
    link_ops: <<T as Sortable>::Link as Link>::LinkOps,
    pointer_ops: DefaultPointerOps<UnsafeRef<T>>,
    _phantom: PhantomData<UnsafeRef<T>>,
}

impl<T> SortedKeyAdapter<T>
where
    T: Sortable,
{
    pub fn new(order: SortOrder) -> Self {
        Self {
            order,
            link_ops: Default::default(),
            pointer_ops: Default::default(),
            _phantom: PhantomData,
        }
    }
}

unsafe impl<T> Send for SortedKeyAdapter<T> where T: Sortable {}
unsafe impl<T> Sync for SortedKeyAdapter<T> where T: Sortable {}

impl<T, L> Clone for SortedKeyAdapter<T>
where
    L: Link,
    T: Sortable<Link = L> + Clone,
{
    fn clone(&self) -> Self {
        let order = self.order;
        SortedKeyAdapter {
            order,
            link_ops: self.link_ops.clone(),
            pointer_ops: self.pointer_ops.clone(),
            _phantom: PhantomData,
        }
    }
}

#[allow(dead_code, unsafe_code)]
unsafe impl<T> Adapter for SortedKeyAdapter<T>
where
    T: Sortable,
{
    // Currently the only sorted collection is RBTree
    type LinkOps = <<T as Sortable>::Link as Link>::LinkOps;
    type PointerOps = DefaultPointerOps<UnsafeRef<T>>;

    #[inline]
    unsafe fn get_value(
        &self,
        link: <Self::LinkOps as LinkOps>::LinkPtr,
    ) -> *const <Self::PointerOps as PointerOps>::Value {
        <T as Sortable>::get_value(link, self.order)
    }

    #[inline]
    unsafe fn get_link(
        &self,
        value: *const <Self::PointerOps as PointerOps>::Value,
    ) -> <Self::LinkOps as LinkOps>::LinkPtr {
        <T as Sortable>::get_link(value, self.order)
    }

    fn link_ops(&self) -> &Self::LinkOps {
        &self.link_ops
    }

    fn link_ops_mut(&mut self) -> &mut Self::LinkOps {
        &mut self.link_ops
    }

    fn pointer_ops(&self) -> &Self::PointerOps {
        &self.pointer_ops
    }
}

impl<'a, T> KeyAdapter<'a> for SortedKeyAdapter<T>
where
    T: Sortable,
{
    type Key = SortKey;

    fn get_key(&self, value: &'a <Self::PointerOps as PointerOps>::Value) -> Self::Key {
        value.sort_key(self.order)
    }
}
