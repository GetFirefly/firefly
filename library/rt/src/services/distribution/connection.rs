use alloc::boxed::Box;
use alloc::sync::Arc;

use firefly_system::sync::Atomic;
use firefly_system::time::MonotonicTime;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListAtomicLink};

use crate::process::link::LinkTree;
use crate::process::monitor::MonitorList;
use crate::process::ProcessList;
use crate::services::registry::WeakAddress;
use crate::term::{atoms, Atom, OpaqueTerm, Port};

/// The connection state of a given node
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeStatus {
    /// The node is not currently connected/unavailable
    Disconnected = 0,
    /// We're currently in the process of connecting to the node
    Pending,
    /// The node is connected and visible
    Visible,
    /// The node is connected, but hidden
    Hidden,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ConnectionError {
    /// Connect failed because the node could not be reached
    Unreachable,
    /// Connect failed because the node cookie was wrong
    Unauthenticated,
    /// Connect failed because the remote node is not allowing connections from us
    Unauthorized,
}

intrusive_adapter!(pub NodeConnectionAdapter = Arc<NodeConnection>: NodeConnection { link: LinkedListAtomicLink });

#[allow(unused)]
pub type NodeConnectionList = LinkedList<NodeConnectionAdapter>;

/// This structure represents the connection backing a [`Node`],
/// and corresponds to `dist_entry_` in `erl_node_tables.h`.
///
/// Every node but the current one is associated with its connection,
/// and the connection holds all of the state for communicating with
/// the node, as well as various bits of node state that are relevant
/// to the connection (e.g. node name and cookie).
///
/// If a connection is lost, the node connection is what tracks the links/monitors
/// to trigger, and attempts to reconnect on failure, triggering any node monitors
/// which are set on this node.
#[allow(unused)]
pub struct NodeConnection {
    link: LinkedListAtomicLink,
    /// Unique identifier for this connection, incremented on every connection
    id: u32,
    pub name: Atom,
    creation: u32,
    input_handler: Atomic<OpaqueTerm>,
    /// The process or port which owns this connection
    ///
    /// If `None`, this connection is unused.
    connection_handler_id: Option<WeakAddress>,
    status: NodeStatus,
    pending_nodedown: bool,
    // This is a reference to a process
    suspended_nodeup: OpaqueTerm,
    flags: u64,
    opts: u32,
    links: LinkTree,
    monitors: MonitorList,
    suspended: ProcessList,
    send: Option<Box<dyn Fn(Arc<Port>, &[u8]) -> u32>>,
}
/// This is safe (for now) because currently NodeConnection is read-only, and in
/// the future we will be making individual fields Sync as we develop the
/// distribution system
unsafe impl Sync for NodeConnection {}
unsafe impl Send for NodeConnection {}
impl NodeConnection {
    const ERTS_DIST_CON_ID_MASK: u32 = 0x00ffffff;

    pub fn new() -> Arc<Self> {
        let connection_id = MonotonicTime::now();
        let connection_id = connection_id.as_u64() & (Self::ERTS_DIST_CON_ID_MASK as u64);

        Arc::new(Self {
            link: LinkedListAtomicLink::new(),
            id: connection_id as u32,
            name: atoms::NoNodeAtNoHost,
            creation: 0,
            input_handler: Atomic::new(OpaqueTerm::NIL),
            connection_handler_id: None,
            status: NodeStatus::Disconnected,
            pending_nodedown: false,
            suspended_nodeup: OpaqueTerm::NONE,
            flags: 0,
            opts: 0,
            links: LinkTree::default(),
            monitors: MonitorList::default(),
            suspended: ProcessList::default(),
            send: None,
        })
    }
}
