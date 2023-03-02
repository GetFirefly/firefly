use alloc::sync::{Arc, Weak};
use core::sync::atomic::Ordering;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{
    KeyAdapter, LinkedList, LinkedListAtomicLink, RBTree, RBTreeAtomicLink,
};

use firefly_system::sync::Atomic;

use crate::services::distribution::NodeConnection;
use crate::services::registry::WeakAddress;
use crate::term::{atoms, Atom, OpaqueTerm, Pid, Reference, ReferenceId, Term, TermFragment};

use super::ProcessId;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum UnaliasMode {
    /// Only an explicit call to `unalias/1` will deactivate alias
    #[default]
    Explicit = 1,
    /// The alias will be automatically deactivated when the monitor is removed;
    /// either by a call to `demonitor/1`, or when a monitor down signal is delivered.
    ///
    /// It is still permitted to call `unalias/1` explicitly.
    Demonitor,
    /// Same as above, but additionally, when a reply message is received via the alias,
    /// the monitor will also be automatically removed. This is useful in client/server
    /// scenarios when a client monitors the server and will get the reply via the alias.
    /// Once the response is received, both the alias and the monitor will be automatically
    /// removed regardless of whether the response is a reply or a monitor down message.
    ///
    /// It is still permitted to call `unalias/1` explicitly. Note however that when doing so,
    /// the monitor will still be left active.
    ReplyDemonitor,
}
impl TryFrom<Atom> for UnaliasMode {
    type Error = ();

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        match value {
            v if v == atoms::Explicit => Ok(Self::Explicit),
            v if v == atoms::Demonitor => Ok(Self::Demonitor),
            v if v == atoms::ReplyDemonitor => Ok(Self::Explicit),
            _ => Err(()),
        }
    }
}

bitflags::bitflags! {
    pub struct MonitorFlags: u16 {
        /// This is a monitor for a pending spawn request
        const SPAWN_PENDING = 1 << 1;
        /// The spawn request should monitor the spawned process
        const SPAWN_MONITOR = 1 << 2;
        /// The spawn request should link to the spawned process
        const SPAWN_LINK = 1 << 3;
        /// The spawn request has been abandoned
        const SPAWN_ABANDONED = 1 << 4;
        /// No spawn reply message will be sent if the spawn operation succeeds
        const SPAWN_NO_REPLY_SUCCESS = 1 << 5;
        /// No spawn reply message will be sent if the spawn operation fails
        const SPAWN_NO_REPLY_ERROR = 1 << 6;
        /// The monitor reference is also a valid alias
        const ALIAS = 1 << 7;
        /// The alias represented by the monitor reference has the `UnaliasMode::Explicit` behavior
        ///
        /// This is the default aliasing behavior.
        ///
        /// Only meaningful when `ALIAS` is set.
        ///
        /// If `ALIAS` is set, and both this and `UNALIAS_REPLY` are not, it implies `UnaliasMode::Demonitor` behavior
        const UNALIAS_EXPLICIT = 1 << 8;
        /// The alias represented by the monitor reference has the `UnaliasMode::ReplyDemonitor` behavior
        ///
        /// Only meaningful when `ALIAS` is set and `UNALIAS_EXPLICIT` is unset.
        const UNALIAS_REPLY = 1 << 9;
        /// If set, the `name_or_tag` field of the monitor info contains a tag, not a name
        const TAG = 1 << 10;

        /// The default alias options, when aliasing is enabled
        const ALIAS_DEFAULT = Self::ALIAS.bits | Self::UNALIAS_EXPLICIT.bits;

        /// Mask for alias-related flags
        const ALIAS_MASK = Self::ALIAS.bits | Self::UNALIAS_EXPLICIT.bits | Self::UNALIAS_REPLY.bits;

        /// Mask for spawn-related flags
        const SPAWN_MASK = Self::SPAWN_PENDING.bits
            | Self::SPAWN_MONITOR.bits
            | Self::SPAWN_LINK.bits
            | Self::SPAWN_ABANDONED.bits
            | Self::SPAWN_NO_REPLY_SUCCESS.bits
            | Self::SPAWN_NO_REPLY_ERROR.bits;
    }
}
impl firefly_system::sync::Atom for MonitorFlags {
    type Repr = u16;

    #[inline(always)]
    fn pack(self) -> Self::Repr {
        self.bits()
    }

    #[inline(always)]
    fn unpack(raw: Self::Repr) -> Self {
        unsafe { Self::from_bits_unchecked(raw) }
    }
}
impl firefly_system::sync::AtomLogic for MonitorFlags {}
impl MonitorFlags {
    /// Returns `None` if the monitor reference is not a valid alias.
    ///
    /// Otherwise, returns an `UnaliasMode` value representing the desired unaliasing behavior.
    pub fn alias(&self) -> Option<UnaliasMode> {
        if !self.contains(Self::ALIAS) {
            return None;
        }
        if self.contains(Self::UNALIAS_EXPLICIT) {
            Some(UnaliasMode::Explicit)
        } else if self.contains(Self::UNALIAS_REPLY) {
            Some(UnaliasMode::ReplyDemonitor)
        } else {
            Some(UnaliasMode::Demonitor)
        }
    }
}
impl core::ops::BitOr<UnaliasMode> for MonitorFlags {
    type Output = MonitorFlags;

    fn bitor(mut self, rhs: UnaliasMode) -> Self::Output {
        self |= rhs;
        self
    }
}
impl core::ops::BitOrAssign<UnaliasMode> for MonitorFlags {
    fn bitor_assign(&mut self, rhs: UnaliasMode) {
        self.remove(Self::ALIAS_MASK);
        *self |= match rhs {
            UnaliasMode::Explicit => Self::ALIAS | Self::UNALIAS_EXPLICIT,
            UnaliasMode::Demonitor => Self::ALIAS,
            UnaliasMode::ReplyDemonitor => Self::ALIAS | Self::UNALIAS_REPLY,
        };
    }
}

intrusive_adapter!(pub MonitorListAdapter = Arc<MonitorEntry>: MonitorEntry { target: LinkedListAtomicLink });
intrusive_adapter!(pub MonitorTreeAdapter = Arc<MonitorEntry>: MonitorEntry { origin: RBTreeAtomicLink });
impl<'a> KeyAdapter<'a> for MonitorTreeAdapter {
    type Key = ReferenceId;

    #[inline]
    fn get_key(&self, entry: &'a MonitorEntry) -> Self::Key {
        entry.key()
    }
}

/// A type alias for an intrusive linked list of [`MonitorEntry`] stored on a target process/port
pub type MonitorList = LinkedList<MonitorListAdapter>;
/// A type alias for an intrusive red-black tree of [`MonitorEntry`] stored on the origin
/// process/port
pub type MonitorTree = RBTree<MonitorTreeAdapter>;
/// A type alias for entries in a [`MonitorTree`]
pub type MonitorTreeEntry<'a> = intrusive_collections::rbtree::Entry<'a, MonitorTreeAdapter>;

/// This struct serves a few different purposes:
///
/// * It represents one end of a monitor, i.e. origin or target
/// * It provides the ability to store one end of a monitor in a linked list or rbtree
/// * It provides context to the monitor, e.g. what flags were set
pub struct MonitorEntry {
    // Used to attach this monitor to the target process
    target: LinkedListAtomicLink,
    // Used to attach this monitor to the origin process
    origin: RBTreeAtomicLink,
    pub flags: Atomic<MonitorFlags>,
    pub monitor: Monitor,
}
unsafe impl Send for MonitorEntry {}
unsafe impl Sync for MonitorEntry {}
impl MonitorEntry {
    pub fn new(monitor: Monitor) -> Arc<Self> {
        Arc::new(Self {
            target: LinkedListAtomicLink::new(),
            origin: RBTreeAtomicLink::new(),
            flags: Atomic::new(MonitorFlags::empty()),
            monitor,
        })
    }

    /// Returns true if this monitor entry is present in the target process monitor list
    #[inline]
    pub fn is_target_linked(&self) -> bool {
        self.target.is_linked()
    }

    /// Returns true if this monitor entry is present in the origin process monitor tree
    #[inline]
    pub fn is_origin_linked(&self) -> bool {
        self.origin.is_linked()
    }

    /// Returns the current set of flags on this monitor
    #[inline]
    pub fn flags(&self) -> MonitorFlags {
        self.flags.load(Ordering::Acquire)
    }

    /// Sets `flags` on this monitor entry
    pub fn set_flags(&self, flags: MonitorFlags) {
        self.flags.fetch_or(flags, Ordering::Release);
    }

    /// Removes `flags` from this monitor entry
    pub fn remove_flags(&self, flags: MonitorFlags) {
        self.flags.fetch_and(!flags, Ordering::Release);
    }

    /// Returns the sender/origin of this monitor
    pub fn origin(&self) -> Option<WeakAddress> {
        match &self.monitor {
            Monitor::LocalProcess { ref origin, .. }
            | Monitor::TimeOffset { ref origin, .. }
            | Monitor::ToExternalProcess { ref origin, .. }
            | Monitor::Node { ref origin, .. }
            | Monitor::Nodes { ref origin, .. }
            | Monitor::Suspend { ref origin, .. }
            | Monitor::Alias { ref origin, .. } => Some((*origin).into()),
            Monitor::LocalPort { ref origin, .. } => Some(origin.clone()),
            Monitor::FromExternalProcess { ref origin, .. } => Some(origin.clone().into()),
            Monitor::Resource { .. } => None,
        }
    }

    /// Returns the target of this monitor
    pub fn target(&self) -> Option<WeakAddress> {
        match &self.monitor {
            Monitor::LocalProcess { ref target, .. }
            | Monitor::FromExternalProcess { ref target, .. }
            | Monitor::Resource { ref target, .. }
            | Monitor::Suspend { ref target, .. } => Some((*target).into()),
            Monitor::Alias { .. }
            | Monitor::Nodes { .. }
            | Monitor::Node { .. }
            | Monitor::TimeOffset { .. } => None,
            Monitor::LocalPort { ref target, .. } => Some(target.clone()),
            Monitor::ToExternalProcess { ref target, .. } => Some(target.clone().into()),
        }
    }

    /// Returns the name associated with this monitor, if applicable, and set
    pub fn name(&self) -> Option<Atom> {
        if self.flags().contains(MonitorFlags::TAG) {
            None
        } else {
            match &self.monitor {
                Monitor::LocalProcess { ref info, .. }
                | Monitor::LocalPort { ref info, .. }
                | Monitor::TimeOffset { ref info, .. } => match info.name_or_tag.term.into() {
                    Term::None => None,
                    Term::Atom(name) => Some(name),
                    _ => unreachable!(),
                },
                Monitor::ToExternalProcess { ref info, .. } => match info.name_or_tag.term.into() {
                    Term::None => None,
                    Term::Atom(name) => Some(name),
                    _ => unreachable!(),
                },
                Monitor::FromExternalProcess { ref info, .. } => match info.name_or_tag.term.into()
                {
                    Term::None => None,
                    Term::Atom(name) => Some(name),
                    _ => unreachable!(),
                },
                Monitor::Alias { .. }
                | Monitor::Resource { .. }
                | Monitor::Node { .. }
                | Monitor::Nodes { .. }
                | Monitor::Suspend { .. } => None,
            }
        }
    }

    pub fn node_name(&self) -> Atom {
        match &self.monitor {
            Monitor::Alias { .. } | Monitor::LocalProcess { .. } | Monitor::LocalPort { .. } => {
                atoms::NoNodeAtNoHost
            }
            Monitor::ToExternalProcess { ref info, .. } => match info.dist.upgrade() {
                None => atoms::NoNodeAtNoHost,
                Some(dist) => dist.name,
            },
            Monitor::FromExternalProcess { ref info, .. } => match info.dist.upgrade() {
                None => atoms::NoNodeAtNoHost,
                Some(dist) => dist.name,
            },
            Monitor::Node { target, .. } => *target,
            _ => atoms::NoNodeAtNoHost,
        }
    }

    /// Returns the custom tag associated with this monitor, if applicable, and set
    pub fn tag(&self) -> Option<OpaqueTerm> {
        if !self.flags().contains(MonitorFlags::TAG) {
            return None;
        }

        let term = match &self.monitor {
            Monitor::LocalProcess { ref info, .. }
            | Monitor::LocalPort { ref info, .. }
            | Monitor::TimeOffset { ref info, .. } => info.name_or_tag.term,
            Monitor::ToExternalProcess { ref info, .. } => info.name_or_tag.term,
            Monitor::FromExternalProcess { ref info, .. } => info.name_or_tag.term,
            Monitor::Resource { .. } => return None,
            Monitor::Node { ref info, .. } | Monitor::Nodes { ref info, .. } => info.tag.term,
            Monitor::Suspend { ref info, .. } => info.tag.term,
            Monitor::Alias { .. } => OpaqueTerm::NONE,
        };

        if term.is_none() {
            None
        } else {
            Some(term)
        }
    }

    #[doc(hidden)]
    pub fn key(&self) -> ReferenceId {
        match &self.monitor {
            Monitor::LocalProcess { ref info, .. }
            | Monitor::LocalPort { ref info, .. }
            | Monitor::TimeOffset { ref info, .. } => info.reference,
            Monitor::ToExternalProcess { ref info, .. } => info.reference,
            Monitor::FromExternalProcess { ref info, .. } => info.reference.id(),
            Monitor::Resource { ref info, .. } => *info,
            Monitor::Node { ref info, .. } | Monitor::Nodes { ref info, .. } => info.reference,
            Monitor::Suspend { ref info, .. } => info.reference,
            Monitor::Alias { ref reference, .. } => reference.id(),
        }
    }
}

/// Represents the various types of monitors supported by the runtime system
///
/// A number of these are deliberately specialized around the type and direction of the monitor,
/// even though they could be considered the same general monitor. Specifically, monitoring local
/// processes/ports differs from monitoring, or being monitored by, external processes/ports. If
/// we can easily differentiate the locality and direction of the monitor, we can optimize local
/// operations more easily.
pub enum Monitor {
    /// A local process (origin) monitors another local process (target)
    ///
    /// Origin part of the monitor is stored in monitor tree of origin process,
    /// and target part of the monitor is stored in monitor list for local targets on the
    /// target process
    LocalProcess {
        origin: ProcessId,
        target: ProcessId,
        info: LocalMonitorInfo,
    },
    /// A local process (origin) monitors a local port (target), or
    /// a local port (origin) monitors a local process (target).
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin process/port
    /// and target part of the monitor is stored in monitor list for local targets on the
    /// target process/port.
    LocalPort {
        origin: WeakAddress,
        target: WeakAddress,
        info: LocalMonitorInfo,
    },
    /// A local process (origin) monitors time offset (target)
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin process
    /// and target part of the monitor is stored in global time offset monitors list
    TimeOffset {
        origin: ProcessId,
        info: LocalMonitorInfo,
    },
    /// A local process (origin) monitors a remote process (target)
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin process and
    /// target part of monitor is stored in `monitors` list of the dist structure
    ToExternalProcess {
        origin: ProcessId,
        target: Pid,
        info: RemoteMonitorInfo,
    },
    /// A remote process (origin) monitors a local process (target)
    ///
    /// If the monitor is by name, the origin part of the monitor is stored in the monitor
    /// tree `origin_named_monitors` in the dist structure. The target part of the monitor is
    /// stored in the monitor tree of the local process
    FromExternalProcess {
        origin: Pid,
        target: ProcessId,
        info: ExternalMonitorInfo,
    },
    /// A NIF resource (origin) monitors a local process (target)
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin resource, and
    /// target part of the monitor is stored in the monitor list for local targets on the
    /// target process.
    Resource {
        origin: Reference,
        target: ProcessId,
        info: ReferenceId,
    },
    /// A local process (origin) monitors a distribution connection (target) via
    /// `erlang:monitor_node/0`
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin process, and target part
    /// of the monitor is stored in `monitors` list of the dist structure.
    Node {
        origin: ProcessId,
        target: Atom,
        info: NodeMonitorInfo,
    },
    /// A local process (origin) monitors all connections (target) via `net_kernel:monitor_nodes/0`
    ///
    /// Origin part of the monitor is stored in the monitor tree of the origin process, and target
    /// part is stored in the global `nodes_monitors` list.
    Nodes {
        origin: ProcessId,
        mask: usize,
        info: NodeMonitorInfo,
    },
    /// A local process (origin) suspends another local process (target)
    ///
    /// Origin part of the monitor is stored in the monitor tree of origin process, and target
    /// part is stored in monitor list for local targets on the target process.
    Suspend {
        origin: ProcessId,
        target: ProcessId,
        info: SuspendMonitorInfo,
    },
    /// A monitor for an alias
    Alias {
        origin: ProcessId,
        reference: Reference,
    },
}

pub struct LocalMonitorInfo {
    pub reference: ReferenceId,
    /// A tag is a term (possibly heap allocated) used in place of the default `Tag`
    /// in a monitor message to a process or port.
    ///
    /// When a monitor message is sent to a process, it looks like so:
    ///
    /// ```erlang
    ///     {'DOWN', MonitorRef, Type, From, Payload}
    /// ```
    ///
    /// Where `'DOWN'` is the tag for the message. When a custom tag is used, it replaces
    /// that default tag with whatever term was given. This is typically used to enable a
    /// specific selective receive transformation that works when a alias/monitor reference
    /// is created just prior to a receive block which uses that reference in its matches.
    /// An example of a custom tag would be `{'DOWN', RequestId}`, where `RequestId` is a
    /// reference. This allows skipping past all of the messages in the queue which occurred
    /// before the reference was created, and thus could not possibly match.
    pub name_or_tag: TermFragment,
}

/// A monitor to a remote process
pub struct RemoteMonitorInfo {
    pub reference: ReferenceId,
    pub name_or_tag: TermFragment,
    pub dist: Weak<NodeConnection>,
}

/// A monitor from a remote process
pub struct ExternalMonitorInfo {
    pub reference: Reference,
    pub name_or_tag: TermFragment,
    /// Pointer to distribution structure, which is not yet defined
    pub dist: Weak<NodeConnection>,
}

/// Metadata required for node monitors
pub struct NodeMonitorInfo {
    /// The reference associated with this monitor
    pub reference: ReferenceId,
    /// Number of invocations to `erlang:monitor_node/0`
    pub reference_count: usize,
    /// If set, uses a custom tag for this monitor
    pub tag: TermFragment,
}

/// Metadata required for suspend monitors
pub struct SuspendMonitorInfo {
    /// The reference associated with this monitor
    pub reference: ReferenceId,
    /// Number of suspends
    pub suspends: u32,
    /// Is the suspend active
    pub active: bool,
    /// If set, uses a custom tag for this monitor
    pub tag: TermFragment,
}
