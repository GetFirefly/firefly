mod connection;
mod node;

pub use self::connection::{ConnectionError, NodeConnection, NodeStatus};
pub use self::node::Node;

use alloc::sync::Arc;
use alloc::{vec, vec::Vec};
use core::ops::Deref;
use core::sync::atomic::{AtomicBool, Ordering};

use firefly_system::sync::{Atomic, OnceLock};

use crate::term::{atoms, Atom};

static DISTRIBUTION: OnceLock<Arc<dyn DistributionService>> = OnceLock::new();

#[derive(Debug, Copy, Clone)]
pub enum DistributionError {
    /// Indicates that distribution could not be started due to invalid or missing configuration
    InvalidOrMissingConfig,
    /// Indicates that distribution could not be started because it is already started
    AlreadyStarted,
    /// Indicates that an operation failed because distribution is not started
    NotStarted,
    /// Indicates that an operation failed because either the current node,
    /// or a given node, is not part of a distributed system.
    NotAlive,
    /// A connection failure occurred
    ConnectionError(ConnectionError),
}
impl From<ConnectionError> for DistributionError {
    #[inline]
    fn from(err: ConnectionError) -> Self {
        Self::ConnectionError(err)
    }
}

/// Initializes distribution using the provided service implementation
///
/// This function may only be called once after the system is started, and must be
/// called before any uses of distribution occur.
pub fn init(dist: Arc<dyn DistributionService>) {
    if let Err(_) = DISTRIBUTION.set(dist) {
        panic!("tried to init the distribution service twice!");
    }
}

/// Starts the distribution service.
///
/// Starting is separate from initialization, and implies that the current node
/// is available to connect to remote nodes (or be connected to). This requires
/// that a valid configuration for the underlying service implementation is present,
/// so this function returns `Ok` if successful, or `Err` if the service could not start.
pub fn start(name: Atom) -> Result<(), DistributionError> {
    with_distribution(move |dist| dist.start(name))
}

/// Stops distribution, disconnecting any connected nodes, and triggering any links/monitors
/// which are still active.
pub fn stop() -> Result<(), DistributionError> {
    with_distribution_started(|dist| dist.stop())
}

/// Returns true if the distribution service is started
///
/// This can be used to implement `erlang:is_alive/0`.
pub fn is_started() -> bool {
    if let Some(dist) = DISTRIBUTION.get() {
        dist.is_started()
    } else {
        false
    }
}

/// Connects to `node` via distribution.
///
/// If the node is already connected, this always returns `Ok`.
///
/// If a connection cannot be established, `Err` is returned with the reason wny.
///
/// NOTE: Distribution must be started to connect nodes.
pub fn connect(node: Atom) -> Result<Arc<Node>, DistributionError> {
    with_distribution_started(move |dist| dist.connect(node))
}

/// Returns a reference to the current node
pub fn current_node() -> Arc<Node> {
    with_distribution(move |dist| dist.current_node())
}

/// Sets the magic cookie of `node` to `cookie`.
///
/// If `node` is the local/current node, then `cookie` is also used as the default
/// cookie for all unknown nodes.
///
/// Returns `Err` if distribution is not started, or `node` is not alive.
pub fn set_cookie(node: Atom, cookie: Atom) -> Result<(), DistributionError> {
    with_distribution_started(move |dist| dist.set_cookie(node, cookie))
}

/// Returns a `Vec` containing all of the currently connected nodes
pub fn list() -> Vec<Arc<Node>> {
    with_distribution(|dist| dist.list())
}

/// Returns a `Vec` containing all the nodes currently in `status`.
pub fn list_by_status(status: NodeStatus) -> Vec<Arc<Node>> {
    with_distribution(|dist| dist.list_by_status(status))
}

#[inline(always)]
fn with_distribution<F, T>(callback: F) -> T
where
    F: FnOnce(&dyn DistributionService) -> T,
{
    let dist = DISTRIBUTION
        .get()
        .expect("distribution has not been initialized!");
    callback(dist.deref())
}

#[inline(always)]
fn with_distribution_started<F, T>(callback: F) -> Result<T, DistributionError>
where
    F: FnOnce(&dyn DistributionService) -> Result<T, DistributionError>,
{
    let dist = DISTRIBUTION.get().ok_or(DistributionError::NotStarted)?;
    callback(dist.deref())
}

// TODO/NOTE:
//
// Here's what we need to do with regards to distribution:
//
// * Define node table, which contains a mapping from (node_name, creation) to Arc<Node>
// * Define distribution table, which contains a mapping from `node_name` to Arc<NodeConnection>,
// the latter is used to represent a connection (potential or otherwise) to a node. `Node` holds
// a `Arc<NodeConnection>` reference for its corresponding connection.
// * The node/distribution table are both initialized with a default entry for the current node,
// but the connection is obviously meaningless in that case, so we'll need some way to indicate that
// the connection is established, but without all of the other bits.
// * This crate should not have any of the connection logic in it, just the high-level types used in
// distribution. We can elide over the details of such things for now.
//
// Here's what remains to do:
//
// * [x] Finish up the representation of links/monitors
// * [ ] Verify signal queueing
// * [ ] Verify timer service/wheel
// * [x] Rename Interpreter to Emulator
// * [x] Update Emulator to account for recent changes
// * [x] Separate bytecode from emulator crate and make it a library
// * [ ] Add new pass to codegen crate which lowers SSA to bytecode, and generates an LLVM module
// containing the bytecode as a global static, and a shim function which invokes the emulator
// main function, passing it a reference to the bytecode static
// * [ ] Ensure the emulator is built as a library which can be linked in as a runtime crate
// * [ ] Modify the driver so that the default compilation path generates the bytecode module
// and links an executable with the emulator
// * [ ] Test the generated executable

pub trait DistributionService: Send + Sync {
    /// Starts distribution using `name` as the local node name.
    ///
    /// Distribution allows the current node to connect to other nodes over the network
    /// as long as they are reachable and have the same cookie value.
    ///
    /// By default, distribution is stopped/disabled, and must be explicitly enabled either
    /// by command-line options, or by request.
    ///
    /// Returns `Ok` if successfully started, or `Err` if unable to start for some reason.
    fn start(&self, name: Atom) -> Result<(), DistributionError>;
    /// Stops distribution, disconnecting any connected nodes, and triggering any links/monitors
    /// which are still active.
    fn stop(&self) -> Result<(), DistributionError>;
    /// Returns true if distribution has been started and is available
    fn is_started(&self) -> bool;
    /// Connects to `node` via distribution.
    ///
    /// If the node is already connected, this always returns `Ok`.
    ///
    /// If a connection cannot be established, `Err` is returned with the reason wny.
    ///
    /// NOTE: Distribution must be started to connect nodes.
    fn connect(&self, node: Atom) -> Result<Arc<Node>, DistributionError>;
    /// Returns the current node
    ///
    /// There is always a node representing the current machine, even if distribution is stopped
    fn current_node(&self) -> Arc<Node>;
    /// Sets the magic cookie to use with `node` to `cookie`
    ///
    /// If `node` is the local/current node, then the default cookie used with all unknown nodes
    /// will be set to `cookie` as well.
    ///
    /// The default cookie for unknown nodes is `nocookie`.
    ///
    /// Returns `Err` if distribution is not started or `node` is not alive
    fn set_cookie(&self, node: Atom, cookie: Atom) -> Result<(), DistributionError>;
    /// Returns a `Vec` containing all of the currently connected nodes
    fn list(&self) -> Vec<Arc<Node>>;
    /// Returns a `Vec` containing all the nodes currently in `status`.
    fn list_by_status(&self, status: NodeStatus) -> Vec<Arc<Node>>;
}

/// A simple distribution service which is not capable of remote connections, it simply
/// provides an implementation of the service interface for use in non-distributed contexts.
pub struct NoDistribution {
    current_node: Arc<Node>,
    default_cookie: Atomic<Atom>,
    started: AtomicBool,
}
unsafe impl Sync for NoDistribution {}
unsafe impl Send for NoDistribution {}
impl NoDistribution {
    pub fn new() -> Arc<Self> {
        let current_node = Arc::new(Node::default());
        Arc::new(Self {
            current_node,
            default_cookie: Atomic::new(atoms::Nocookie),
            started: AtomicBool::new(false),
        })
    }
}
impl DistributionService for NoDistribution {
    fn start(&self, name: Atom) -> Result<(), DistributionError> {
        unsafe {
            self.current_node.set_name(name);
        }
        match self
            .started
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => Ok(()),
            Err(_) => unsafe {
                self.current_node.unset_name();
                Err(DistributionError::AlreadyStarted)
            },
        }
    }

    fn stop(&self) -> Result<(), DistributionError> {
        match self
            .started
            .compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => {
                unsafe {
                    self.current_node.unset_name();
                }
                Ok(())
            }
            Err(_) => Err(DistributionError::NotStarted),
        }
    }

    #[inline]
    fn is_started(&self) -> bool {
        self.started.load(Ordering::Relaxed)
    }

    fn connect(&self, _node: Atom) -> Result<Arc<Node>, DistributionError> {
        Err(ConnectionError::Unreachable.into())
    }

    fn current_node(&self) -> Arc<Node> {
        self.current_node.clone()
    }

    fn set_cookie(&self, node: Atom, cookie: Atom) -> Result<(), DistributionError> {
        if self.current_node.name() == node {
            self.current_node.set_cookie(cookie);
            self.default_cookie.store(cookie, Ordering::Relaxed);
            Ok(())
        } else {
            Err(DistributionError::NotAlive)
        }
    }

    #[inline]
    fn list(&self) -> Vec<Arc<Node>> {
        vec![self.current_node.clone()]
    }

    fn list_by_status(&self, status: NodeStatus) -> Vec<Arc<Node>> {
        match status {
            NodeStatus::Disconnected | NodeStatus::Pending => vec![],
            NodeStatus::Visible | NodeStatus::Hidden => self.list(),
        }
    }
}
