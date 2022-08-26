use core::hash::{Hash, Hasher};
use core::ptr::NonNull;
use core::sync::atomic::AtomicPtr;

use super::{atom::AtomData, Atom};

#[repr(C)]
#[derive(Debug)]
pub struct Node {
    id: usize,
    name: AtomicPtr<AtomData>,
    creation: u32,
}

impl Node {
    /// Creates a new Node from the given metadata
    pub fn new(id: usize, name: Atom, creation: u32) -> Self {
        Self {
            id,
            name: AtomicPtr::new(unsafe { name.as_ptr() as *mut AtomData }),
            creation,
        }
    }

    /// Returns the numeric identifier associated with this node
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name of this node as an atom, if one was set
    pub fn name(&self) -> Option<Atom> {
        use core::sync::atomic::Ordering;

        NonNull::new(self.name.load(Ordering::Relaxed)).map(|ptr| ptr.into())
    }

    /// Returns the creation time of this node
    pub fn creation(&self) -> u32 {
        self.creation
    }
}

impl Eq for Node {}
impl crate::cmp::ExactEq for Node {}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Only the node identifier is considered in comparisons/hashing, as the
        // other values are sensitive to transient environmental conditions, and
        // the identity of the node must remain consistent across such changes
        self.id.hash(state);
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
