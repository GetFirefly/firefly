//! The registry service is used to track spawned processes and ports globally, as
//! well as registered names assigned to them.
//!
//! Any interaction with other processes or ports, particularly through process and port
//! identifiers, typically goes through the registry. The registry is what allows us to take
//! one of those identifiers and convert it into a reference to the underlying entity in memory
//! in order to perform operations against it.
//!
//! This module provides a variety of convenience functions for interacting with the registry, but
//! you can also take a look at the [`Registry`] struct docs as well (in the std implementation),
//! which explains the semantics of how those functions work.

cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        #[path = "std.rs"]
        mod imp;
    } else {
        #[path = "no_std.rs"]
        mod imp;
    }
}

pub use self::imp::*;

use alloc::sync::{Arc, Weak};
use core::hash::{Hash, Hasher};
use core::ptr;

use firefly_system::sync::OnceLock;

use crate::error::ExceptionFlags;
use crate::function::ErlangResult;
use crate::gc::Gc;
use crate::process::{Process, ProcessId, ProcessLock};
use crate::term::{atoms, Atom, Cons, OpaqueTerm, Pid, Port, PortId, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationError {
    AlreadyRegistered,
}

/// An [`Address`] represents an identifier which can be used to uniquely identify
/// a single process or port system-wide.
///
/// An address, with the exception of the `Name` variant, also represents a weak
/// reference to the corresponding process or port to which it refers. This means a
/// process or port's memory lives until all addresses which refer to it are dropped.
///
/// The process or port designated by an address may exit or be killed while the address
/// is still in use. For this reason, the references held by an address are weak, and require
/// upgrading to use. This allows us to easily tell when the address is no longer usable.
#[derive(Clone)]
pub enum Address {
    /// The address is the registered name of a process or port.
    ///
    /// This address type is a bit unique, in that it only becomes unusable if there is
    /// no registrant with the given name at the point at which we try to resolve it
    Name(Atom),
    /// A weak reference to a specific process instance.
    ///
    /// If the process exits, this address is no longer usable
    Process(Weak<Process>),
    /// A weak reference to a specific port instance.
    ///
    /// If the port exits, this address is no longer usable
    Port(Weak<Port>),
}
impl Address {
    /// Attempts to resolve this address to a live registration
    ///
    /// Returns `None` if no such process/port exists, or has exited
    pub fn try_resolve(&self) -> Option<Registrant> {
        match self {
            Self::Name(name) => get_by_name(*name),
            Self::Process(weak) => weak.upgrade().map(Registrant::Process),
            Self::Port(weak) => weak.upgrade().map(Registrant::Port),
        }
    }
}
impl From<Atom> for Address {
    fn from(name: Atom) -> Self {
        Self::Name(name)
    }
}

/// A [`WeakAddress`] is an alternative to [`Address`] which can also be used to uniquely
/// identify a single process or port; but unlike [`Address`], no reference to the process
/// or port memory is held, allowing that memory to be freed even if there are outstanding
/// [`WeakAddress`] references.
///
/// The downside is that the address is more expensive to resolve when the time comes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeakAddress {
    /// Sometimes the runtime system itself needs to act as a sender, without a corresponding
    /// process identifier. This is used in those circumstances.
    System,
    /// A named process (or port), resolved when needed
    Name(Atom),
    /// A process identifier
    Process(Pid),
    /// A port identifier
    Port(PortId),
}
impl WeakAddress {
    /// Returns true if this address refers to the runtime system itself
    #[inline]
    pub fn is_system(&self) -> bool {
        match self {
            Self::System => true,
            _ => false,
        }
    }

    /// Attempts to resolve this address to a live registration
    ///
    /// Returns `None` if no such process/port exists, or has exited
    pub fn try_resolve(&self) -> Option<Registrant> {
        match self {
            Self::System => None,
            Self::Name(name) => get_by_name(*name),
            Self::Process(pid) if pid.is_local() => {
                get_by_process_id(pid.id()).map(Registrant::Process)
            }
            Self::Port(id) => get_by_port_id(*id).map(Registrant::Port),
            // TODO: Need to implement resolution when support for distribution is implemented
            Self::Process(_) => None,
        }
    }
}
impl Hash for WeakAddress {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::System => atoms::System.hash(state),
            Self::Name(name) => name.hash(state),
            Self::Process(pid) => pid.hash(state),
            Self::Port(port) => port.hash(state),
        }
    }
}
impl TryFrom<OpaqueTerm> for WeakAddress {
    type Error = ();

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        let term: Term = term.into();
        term.try_into()
    }
}
impl TryFrom<Term> for WeakAddress {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Atom(a) if a == atoms::System => Ok(Self::System),
            Term::Atom(name) => Ok(Self::Name(name)),
            Term::Pid(pid) => Ok(Self::Process((&*pid).clone())),
            Term::Port(port) => Ok(Self::Port(port.id())),
            _other => Err(()),
        }
    }
}
impl From<Atom> for WeakAddress {
    fn from(name: Atom) -> Self {
        if name == atoms::System {
            Self::System
        } else {
            Self::Name(name)
        }
    }
}
impl From<Pid> for WeakAddress {
    fn from(pid: Pid) -> Self {
        Self::Process(pid)
    }
}
impl From<ProcessId> for WeakAddress {
    fn from(id: ProcessId) -> Self {
        Self::Process(Pid::new_local(id))
    }
}
impl From<PortId> for WeakAddress {
    fn from(port_id: PortId) -> Self {
        Self::Port(port_id)
    }
}
impl PartialEq<Process> for WeakAddress {
    fn eq(&self, other: &Process) -> bool {
        match self {
            Self::Process(pid) if pid.is_local() => pid.id().eq(&other.id()),
            _ => false,
        }
    }
}

#[cfg_attr(test, derive(Debug))]
pub enum Registrant {
    Process(Arc<Process>),
    Port(Arc<Port>),
}
impl Registrant {
    /// Downgrades this to a [`WeakRegistrant`]
    pub fn downgrade(&self) -> WeakRegistrant {
        match self {
            Self::Process(ref process) => WeakRegistrant::Process(Arc::downgrade(process)),
            Self::Port(ref port) => WeakRegistrant::Port(Arc::downgrade(port)),
        }
    }

    /// Calls `register_name` on the underlying entity
    ///
    /// NOTE: This is not the same as calling `register_name` on the registry, this only sets
    /// the registered name in the `Process` or `Port` struct.
    fn register_name(&self, name: Atom) -> Result<(), Atom> {
        match self {
            Self::Process(process) => process.register_name(name),
            Self::Port(port) => port.register_name(name),
        }
    }

    /// Calls `unregister_name` on the underlying entity
    ///
    /// NOTE: This is not the same as calling `unregister_name` on the registry, this only removes
    /// a registered name saved in the `Process` or `Port` struct.
    fn unregister_name(&self) -> Result<(), ()> {
        match self {
            Self::Process(process) => process.unregister_name(),
            Self::Port(port) => port.unregister_name(),
        }
    }

    fn registered_name(&self) -> Option<Atom> {
        match self {
            Self::Process(process) => process.registered_name(),
            Self::Port(port) => port.registered_name(),
        }
    }
}
impl PartialEq<WeakRegistrant> for Registrant {
    #[inline]
    fn eq(&self, other: &WeakRegistrant) -> bool {
        match (self, other) {
            (Self::Process(a), WeakRegistrant::Process(b)) => {
                ptr::eq(Arc::as_ptr(a), Weak::as_ptr(b))
            }
            (Self::Port(a), WeakRegistrant::Port(b)) => ptr::eq(Arc::as_ptr(a), Weak::as_ptr(b)),
            _ => false,
        }
    }
}
impl PartialEq for Registrant {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Process(a), Self::Process(b)) => Arc::ptr_eq(a, b),
            (Self::Port(a), Self::Port(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
impl From<Arc<Process>> for Registrant {
    fn from(process: Arc<Process>) -> Self {
        Self::Process(process)
    }
}
impl From<Arc<Port>> for Registrant {
    fn from(port: Arc<Port>) -> Self {
        Self::Port(port)
    }
}
impl Into<WeakRegistrant> for Registrant {
    fn into(self) -> WeakRegistrant {
        match self {
            Self::Process(process) => WeakRegistrant::Process(Arc::downgrade(&process)),
            Self::Port(port) => WeakRegistrant::Port(Arc::downgrade(&port)),
        }
    }
}

/// A weak version of [`Registrant`], which holds weak references rather than strong references
/// to the underlying process or port. This should be used in any situation where a strong reference
/// isn't needed, or we want to defer strengthening it until later.
#[cfg_attr(test, derive(Debug))]
#[derive(Clone)]
pub enum WeakRegistrant {
    Process(Weak<Process>),
    Port(Weak<Port>),
}
impl WeakRegistrant {
    /// Upgrade to a [`Registrant`] value
    ///
    /// If this returns `None`, it means the underlying entity (process/port) is gone.
    pub fn upgrade(&self) -> Option<Registrant> {
        match self {
            Self::Process(weak) => weak.upgrade().map(Registrant::Process),
            Self::Port(weak) => weak.upgrade().map(Registrant::Port),
        }
    }
}
impl PartialEq<Registrant> for WeakRegistrant {
    #[inline]
    fn eq(&self, other: &Registrant) -> bool {
        match (self, other) {
            (Self::Process(a), Registrant::Process(b)) => ptr::eq(Weak::as_ptr(a), Arc::as_ptr(b)),
            (Self::Port(a), Registrant::Port(b)) => ptr::eq(Weak::as_ptr(a), Arc::as_ptr(b)),
            _ => false,
        }
    }
}
impl PartialEq for WeakRegistrant {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Process(a), Self::Process(b)) => Weak::ptr_eq(a, b),
            (Self::Port(a), Self::Port(b)) => Weak::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// The global registry instance, which is initialized on first use
///
/// The registry itself is `Send` and `Sync`, this simply ensures that it only ever
/// gets initialized exactly once.
static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// Get a reference to the local process registered to `name`
pub fn get_by_name(name: Atom) -> Option<Registrant> {
    with_name_table(|registry, guard| registry.get_by_name(name, guard))
}

/// Get a reference to the local process assigned to `id`
pub fn get_by_process_id(id: ProcessId) -> Option<Arc<Process>> {
    with_process_table(|registry, guard| registry.get_by_process_id(id, guard))
}

/// Get a reference to the process with the given `pid`
#[inline]
pub fn get_by_pid(id: &Pid) -> Option<Arc<Process>> {
    get_by_process_id(id.id())
}

/// Get a reference to the local port assigned to `id`
pub fn get_by_port_id(id: PortId) -> Option<Arc<Port>> {
    with_port_table(|registry, guard| registry.get_by_port_id(id, guard))
}

/// Inserts a process in the registry
///
/// This function will panic if the registry already contains a registration for the same pid
pub fn register_process(process: Arc<Process>) {
    with_process_table(|registry, guard| registry.register_process(process, guard))
}

/// Removes a process from the registry
pub fn unregister_process(pid: ProcessId) -> Option<Arc<Process>> {
    with_process_table(|registry, guard| registry.unregister_process(pid, guard))
}

/// Inserts a port in the registry
///
/// This function will panic if the registry already contains a registration for the same port id
pub fn register_port(port: Arc<Port>) {
    with_port_table(|registry, guard| registry.register_port(port, guard))
}

/// Removes a port from the registry
pub fn unregister_port(id: PortId) -> Option<Arc<Port>> {
    with_port_table(|registry, guard| registry.unregister_port(id, guard))
}

/// Registers `name` to `process`
///
/// Returns a boolean indicating whether or not the registration attempt succeeded
pub fn register_name(name: Atom, to: Registrant) -> Result<(), RegistrationError> {
    with_name_table(|registry, guard| registry.register_name(name, to, guard))
}

/// Removes any existing registration for `registrant`
pub fn unregister_name(name: Registrant) {
    with_name_table(|registry, guard| registry.unregister_name(name, guard))
}

/// Registers `name` to the process or port referenced by `id`.
///
/// Returns `true` if the registration was successful.
///
/// Raises `badarg` if:
///
/// * The name is invalid (not an atom, or the atom 'undefined')
/// * The process/port does not exist
/// * The process/port already have a registered name
/// * The name is already registered to someone else
#[export_name = "erlang:register/2"]
pub extern "C-unwind" fn register2(
    process: &mut ProcessLock,
    name: OpaqueTerm,
    id: OpaqueTerm,
) -> ErlangResult {
    if !name.is_atom() || name == atoms::Undefined {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = name;
        process.exception_info.trace = None;
        return ErlangResult::Err;
    }
    let name = name.as_atom();

    match id.into() {
        Term::Pid(pid) => {
            if let Some(p) = get_by_pid(&pid) {
                if register_name(name, p.into()).is_ok() {
                    return ErlangResult::Ok(true.into());
                }
            }
        }
        Term::Port(port) => {
            if let Some(p) = get_by_port_id(port.id()) {
                if register_name(name, p.into()).is_ok() {
                    return ErlangResult::Ok(true.into());
                }
            }
        }
        _ => (),
    }
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Badarg.into();
    process.exception_info.value = id;
    process.exception_info.trace = None;
    ErlangResult::Err
}

/// Unregisters `name` from whichever registrant it belongs to.
///
/// Returns `true` if the name was registered.
///
/// Raises `badarg` if the name was not registered.
#[export_name = "erlang:unregister/1"]
pub extern "C-unwind" fn unregister1(process: &mut ProcessLock, name: OpaqueTerm) -> ErlangResult {
    if !name.is_atom() {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = name;
        process.exception_info.trace = None;
        return ErlangResult::Err;
    }
    let name = name.as_atom();

    let exists = with_name_table(|registry, guard| {
        if let Some(registrant) = registry.get_by_name(name, guard) {
            registry.unregister_name(registrant, guard);
            true
        } else {
            false
        }
    });

    if exists {
        ErlangResult::Ok(true.into())
    } else {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = name.into();
        process.exception_info.trace = None;
        ErlangResult::Err
    }
}

/// Returns the pid or port with name `name`
#[export_name = "erlang:whereis/1"]
pub extern "C-unwind" fn whereis(process: &mut ProcessLock, name: OpaqueTerm) -> ErlangResult {
    use crate::gc;

    if let Term::Atom(name) = name.into() {
        match get_by_name(name) {
            None => ErlangResult::Ok(atoms::Undefined.into()),
            Some(Registrant::Process(found)) => loop {
                match Gc::new_uninit_in(process) {
                    Ok(mut empty) => unsafe {
                        empty.write(found.pid());
                        return ErlangResult::Ok(empty.assume_init().into());
                    },
                    Err(_) => {
                        assert!(gc::garbage_collect(process, Default::default()).is_ok());
                    }
                }
            },
            Some(Registrant::Port(port)) => ErlangResult::Ok(port.into()),
        }
    } else {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = name;
        process.exception_info.trace = None;
        ErlangResult::Err
    }
}

/// Produces a list of registered names as a term
#[export_name = "erlang:registered/0"]
pub extern "C-unwind" fn registered(process: &mut ProcessLock) -> ErlangResult {
    use crate::gc;
    use crate::term::LayoutBuilder;
    use firefly_alloc::heap::Heap;

    // Get the number of entries in the name table
    let len = with_name_table(|registry, guard| registry.registered_names(guard));

    // Pad with room for extras in case any new registered names appear while walking the table
    let mut builder = LayoutBuilder::new();
    builder.build_list(len + 5);
    let layout = builder.finish();

    if process.heap.heap_available() < layout.size() {
        process.gc_needed = layout.size();
        assert!(gc::garbage_collect(process, Default::default()).is_ok());
    }

    // Build the list on the heap, panicking if for some reason we run out of heap during this
    with_name_table(|registry, guard| {
        let mut tail = OpaqueTerm::NIL;
        for (name, _) in registry.names(guard) {
            let head: OpaqueTerm = name.into();

            let list = Cons::new_in(Cons { head, tail }, process).unwrap();
            tail = list.into();
        }
        Ok(tail)
    })
    .into()
}

/// This function lets you combine multiple operations against the registry by
/// providing a callback which takes a registry reference and returns some value
/// when done.
///
/// Many of the convenience functions above are built on this, but if you intend
/// to perform multiple operations in quick succession, it is recommended you use
/// this instead, and take a guard which keeps the registry stable while you interact
/// with it.
#[inline]
pub fn with_registry<F, T>(f: F) -> T
where
    F: FnOnce(&Registry) -> T,
{
    let registry = REGISTRY.get_or_init(|| Registry::default());
    f(registry)
}

fn with_process_table<F, T>(f: F) -> T
where
    F: FnOnce(&Registry, &ProcessTableGuard<'_>) -> T,
{
    with_registry(move |registry| {
        let guard = registry.process_table_guard();
        f(registry, &guard)
    })
}

fn with_port_table<F, T>(f: F) -> T
where
    F: FnOnce(&Registry, &PortTableGuard<'_>) -> T,
{
    with_registry(move |registry| {
        let guard = registry.port_table_guard();
        f(registry, &guard)
    })
}

fn with_name_table<F, T>(f: F) -> T
where
    F: FnOnce(&Registry, &NameTableGuard<'_>) -> T,
{
    with_registry(move |registry| {
        let guard = registry.name_table_guard();
        f(registry, &guard)
    })
}
