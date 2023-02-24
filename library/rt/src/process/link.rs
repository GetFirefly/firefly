use alloc::alloc::Global;
use alloc::sync::Arc;
use core::num::NonZeroU64;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use rustc_hash::FxHasher;

use crate::services::registry::WeakAddress;
use crate::term::Pid;

use super::ProcessId;

type HashBuilder = core::hash::BuildHasherDefault<FxHasher>;
type HashMap<K, V> = hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;

pub type LinkTreeEntry<'a, 'b> = hashbrown::hash_map::EntryRef<
    'a,
    'b,
    WeakAddress,
    WeakAddress,
    Arc<LinkEntry>,
    HashBuilder,
    Global,
>;

pub type OccupiedEntry<'a, 'b> = hashbrown::hash_map::OccupiedEntryRef<
    'a,
    'b,
    WeakAddress,
    WeakAddress,
    Arc<LinkEntry>,
    HashBuilder,
    Global,
>;
pub type VacantEntry<'a, 'b> = hashbrown::hash_map::VacantEntryRef<
    'a,
    'b,
    WeakAddress,
    WeakAddress,
    Arc<LinkEntry>,
    HashBuilder,
    Global,
>;

/// Represents the set of links a process/port has established.
///
/// Since links are bidirectional, this tree holds links regardless of
/// which direction initiated the link, unlike monitors. However it is still
/// possible to determine which direction a given link was initiated.
#[derive(Default)]
pub struct LinkTree(HashMap<WeakAddress, Arc<LinkEntry>>);
impl LinkTree {
    /// Insert a new link entry for a process/port we're linking to.
    ///
    /// If the given link is a duplicate, this function returns the input entry wrapped in `Err`,
    /// otherwise it returns `Ok`.
    ///
    /// This is called by the sender/origin of a link, and is keyed by the target address of the link.
    pub fn link(&mut self, link: Arc<LinkEntry>) -> Result<(), Arc<LinkEntry>> {
        use hashbrown::hash_map::Entry;

        let ptr = self as *mut LinkTree;
        match self.0.entry(link.target()) {
            Entry::Occupied(_) => Err(link),
            Entry::Vacant(entry) => {
                link.origin.store(ptr, Ordering::Release);
                entry.insert(link);
                Ok(())
            }
        }
    }

    /// Insert a new link entry for a process/port we're being linked by.
    ///
    /// If the given link is a duplicate, this function returns the input entry wrapped in `Err`,
    /// otherwise it returns `Ok`.
    ///
    /// This is called by the receiver/target of a link, and is keyed by the origin address of the link.
    pub fn linked_by(&mut self, link: Arc<LinkEntry>) -> Result<(), Arc<LinkEntry>> {
        use hashbrown::hash_map::Entry;

        let ptr = self as *mut LinkTree;
        match self.0.entry(link.origin()) {
            Entry::Occupied(_) => Err(link),
            Entry::Vacant(entry) => {
                link.target.store(ptr, Ordering::Release);
                entry.insert(link);
                Ok(())
            }
        }
    }

    /// Returns true if the process/port represented by `addr` is linked in this tree
    #[inline]
    pub fn is_linked(&self, addr: &WeakAddress) -> bool {
        self.0.contains_key(&addr)
    }

    /// Gets the link entry corresponding to `addr`
    #[inline]
    pub fn get(&self, addr: &WeakAddress) -> Option<&Arc<LinkEntry>> {
        self.0.get(addr)
    }

    /// Gets the underlying `HashMap` entry for the link corresponding to `addr`
    #[inline]
    pub fn entry<'a, 'b>(&'a mut self, addr: &'b WeakAddress) -> LinkTreeEntry<'a, 'b> {
        self.0.entry_ref(addr)
    }

    /// Removes the link entry corresponding to `addr` from this tree
    pub fn unlink(&mut self, addr: &WeakAddress) -> Option<Arc<LinkEntry>> {
        let link = self.0.remove(addr)?;
        let ptr = self as *mut LinkTree;
        let is_origin = link
            .origin
            .compare_exchange(ptr, ptr::null_mut(), Ordering::Release, Ordering::Relaxed)
            .is_ok();
        let is_target = link
            .target
            .compare_exchange(ptr, ptr::null_mut(), Ordering::Release, Ordering::Relaxed)
            .is_ok();
        assert!(is_origin || is_target);
        Some(link)
    }

    /// Takes the internal `HashMap` of this link tree, replacing it with an empty one.
    #[inline]
    pub fn take(&mut self) -> HashMap<WeakAddress, Arc<LinkEntry>> {
        core::mem::take(&mut self.0)
    }
}

/// This struct represents both sides of a link as well as the link data
///
/// Both the origin and target of a link carry the same entry in their link trees, but
/// manage the `target` and `origin` pointers based on which end of the link they represent.
///
/// The owner of a link tree can determine which end of the link an entry represents by checking
/// the `target` or `origin` pointer against a pointer to the link tree it owns; the entry is present
/// in that tree if its pointer is present in one of those fields.
pub struct LinkEntry {
    target: AtomicPtr<LinkTree>,
    origin: AtomicPtr<LinkTree>,
    unlinking: AtomicU64,
    pub link: Link,
}
impl LinkEntry {
    /// Create a new link entry for use with in the link list/tree of a pair of processes/ports
    pub fn new(link: Link) -> Arc<Self> {
        Arc::new(Self {
            target: AtomicPtr::new(ptr::null_mut()),
            origin: AtomicPtr::new(ptr::null_mut()),
            unlinking: AtomicU64::new(0),
            link,
        })
    }

    /// Returns true if this link entry is present in the target process link list
    #[inline]
    pub fn is_target_linked(&self) -> bool {
        !self.target.load(Ordering::Acquire).is_null()
    }

    /// Returns true if this link entry is present in the origin process link tree
    #[inline]
    pub fn is_origin_linked(&self) -> bool {
        !self.origin.load(Ordering::Acquire).is_null()
    }

    /// Returns the unique unlink identifier associated with this link entry
    ///
    /// Returns `None` if there is no unlink in progress
    #[inline]
    pub fn unlinking(&self) -> Option<NonZeroU64> {
        NonZeroU64::new(self.unlinking.load(Ordering::Acquire))
    }

    /// Sets the unlinking id for the underlying link
    ///
    /// Returns `false` if the id is already set
    #[inline]
    pub fn set_unlinking(&self, id: u64) -> bool {
        self.unlinking
            .compare_exchange(0, id, Ordering::Release, Ordering::Acquire)
            .is_ok()
    }

    /// Returns a `WeakAddress` corresponding to the origin process/port of this link
    pub fn origin(&self) -> WeakAddress {
        match &self.link {
            Link::LocalProcess { ref origin, .. } | Link::ToExternalProcess { ref origin, .. } => {
                (*origin).into()
            }
            Link::LocalPort { ref origin, .. } => origin.clone(),
            Link::FromExternalProcess { ref origin, .. } => origin.clone().into(),
        }
    }

    /// Returns a `WeakAddress` corresponding to the target process/port of this link
    pub fn target(&self) -> WeakAddress {
        match &self.link {
            Link::LocalProcess { ref target, .. }
            | Link::FromExternalProcess { ref target, .. } => (*target).into(),
            Link::LocalPort { ref target, .. } => target.clone(),
            Link::ToExternalProcess { ref target, .. } => target.clone().into(),
        }
    }
}

/// Represents the various types of links supported by the runtime system.
///
/// This overlaps to some degree with [`Monitor`], but we've deliberately separated
/// the two types for type safety reasons. If in the future we need to mix the two
/// together for some reason, I'm sure we can find a way.
pub enum Link {
    /// A local process (origin) links to another local process (target)
    ///
    /// Both origin and target have their respective partner in their link tree
    LocalProcess {
        origin: ProcessId,
        target: ProcessId,
    },
    /// A local process/port (origin) links to another local process/port (target)
    ///
    /// Both origin and target have their respective partner in their link tree
    LocalPort {
        origin: WeakAddress,
        target: WeakAddress,
    },
    /// A link between a local process (origin) and a remote process (target)
    ///
    /// The local process has the remote one in its link tree.
    /// The remote part of the link is stored in the `links` list of the dist structure
    ToExternalProcess { origin: ProcessId, target: Pid },
    /// A link between a remote process (origin) and a local process (target)
    ///
    /// The local process has the remote one in its link tree.
    /// The remote part of the link is stored in the `links` list of the dist structure
    FromExternalProcess { origin: Pid, target: ProcessId },
}
