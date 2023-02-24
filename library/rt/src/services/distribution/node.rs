use alloc::sync::Arc;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::sync::atomic::Ordering;

use firefly_system::sync::Atomic;

use crate::term::{atoms, Atom};

use super::NodeConnection;

/// Represents a node in the distribution subsystem
///
/// By default, distribution is not started, and the default node is `nonode@nohost`.
pub struct Node {
    id: usize,
    name: Atomic<Atom>,
    cookie: Atomic<Atom>,
    creation: u32,
    /// This is only ever `None` for the current node we're on
    connection: Option<Arc<NodeConnection>>,
}
impl Default for Node {
    fn default() -> Self {
        Self {
            id: 0,
            name: Atomic::new(atoms::NoNodeAtNoHost),
            cookie: Atomic::new(atoms::Nocookie),
            creation: 0,
            connection: None,
        }
    }
}
impl Node {
    /// Creates a new Node from the given metadata
    pub fn new(id: usize, name: Atom, cookie: Atom, creation: u32) -> Self {
        Self {
            id,
            name: Atomic::new(name),
            cookie: Atomic::new(cookie),
            creation,
            connection: None,
        }
    }

    /// Returns the numeric identifier associated with this node
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name of this node as an atom
    pub fn name(&self) -> Atom {
        self.name.load(Ordering::Relaxed)
    }

    /// Returns the magic cookie to use with this node
    pub fn cookie(&self) -> Atom {
        self.cookie.load(Ordering::Relaxed)
    }

    /// Sets the magic cookie to use when connecting with this node
    pub fn set_cookie(&self, cookie: Atom) {
        self.cookie.store(cookie, Ordering::Relaxed)
    }

    /// Changes the name of this node.
    ///
    /// This is only valid for the current node, calling this function on any other node will panic.
    ///
    /// This will also panic if the current name is set to anything other than `nonode@nohost`, as the
    /// node name cannot be changed once distribution is started. You must first explicitly unset the
    /// current name with `unset_name`, then call this function.
    ///
    /// # SAFETY
    ///
    /// This function may only be called by implementations of `DistributionService` when handling
    /// requests to set the local node name prior to starting distribution. This must never be called
    /// once distribution is started, as it will cause conflicts with connections to other nodes which
    /// will not be aware of the name change.
    pub unsafe fn set_name(&self, name: Atom) {
        assert!(self.connection.is_none());
        assert!(self
            .name
            .compare_exchange(
                atoms::NoNodeAtNoHost,
                name,
                Ordering::Relaxed,
                Ordering::Relaxed
            )
            .is_ok());
    }

    /// Changes the name of this node to `nonode@nohost`
    ///
    /// This is only valid for the current node, calling this function on any other node will panic.
    ///
    /// # SAFETY
    ///
    /// This function may only be called by implementations of `DistributionService` after
    /// distribution has been stopped and it is safe to modify the name of the local node.
    pub unsafe fn unset_name(&self) {
        assert!(self.connection.is_none());
        self.name.store(atoms::NoNodeAtNoHost, Ordering::Relaxed)
    }

    /// Returns the creation time of this node
    pub fn creation(&self) -> u32 {
        self.creation
    }
}
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.name, f)
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
        self.name()
            .cmp(&other.name())
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
