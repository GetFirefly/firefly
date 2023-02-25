use alloc::sync::Arc;
use core::intrinsics::unlikely;

use rustc_hash::FxHasher;

use crate::process::{Process, ProcessId};
use crate::term::{Atom, Port, PortId};

use super::{Registrant, RegistrationError, WeakRegistrant};

type HashMap<K, V> = flurry::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;

/// A guard reference for the process table
pub type ProcessTableGuard<'a> = flurry::Guard<'a>;

/// A guard reference for the port table
pub type PortTableGuard<'a> = flurry::Guard<'a>;

/// A guard reference for the registered names table
pub type NameTableGuard<'a> = flurry::Guard<'a>;

/// An iterator over entries in the registered names table
pub type RegisteredNameIter<'a> = flurry::iter::Iter<'a, Atom, WeakRegistrant>;

/// The registry maintains a database of processes and ports, and names registered to them.
///
/// A process or port is not considered valid until it has an entry in the registry.
/// This is because the registry is the only means by which a process or port identifier can be
/// converted into a reference to the actual process/port structure in memory.
///
/// The registry holds a strong reference to all processes/ports it has registered. This ensures
/// that interacting with a registered process is always safe. The exception to this is the names
/// table, which contains weak references. This is because registered names must be associated with
/// registered processes/ports, so dual strong references would be redundant. Registered names are
/// typically unregistered on exit of a process/port anyway, so this setup allows a process to be
/// terminated and for the name registration to be cleaned up lazily.
///
/// A process or port which is exiting will have an entry in the registry during the initial phase
/// of the exit, where a process/port is terminating but technically still alive while it performs
/// last minute work on its way out. During the final phase of cleanup though, its entry will be removed,
/// ideally when that happens the last remaining reference to that process/port is the one returned from
/// the registry itself.
///
/// This registry implementation is built on [`flurry::HashMap`] for each table, which provides us with some
/// nice guarantees when it comes to traversing the table without unnecessarily competing with other threads
/// trying to register things.
#[derive(Default)]
pub struct Registry {
    processes: HashMap<ProcessId, Arc<Process>>,
    ports: HashMap<PortId, Arc<Port>>,
    names: HashMap<Atom, WeakRegistrant>,
}
impl Registry {
    /// Return a guard which can be used to access the process table safely
    ///
    /// Holding a guard to any of the tables prevents collection of garbage generated
    /// by the underlying table, but provides a stable snapshot of the table to work
    /// against. You should try to acquire a guard once and use it across multiple ops
    /// rather than getting one each time - but you should balance this with the concern
    /// around garbage collection.
    #[inline]
    pub fn process_table_guard(&self) -> ProcessTableGuard<'_> {
        self.processes.guard()
    }

    /// Return a guard which can be used to access the port table safely
    ///
    /// See the note on `process_table_guard`, for notes on using guards in general.
    #[inline]
    pub fn port_table_guard(&self) -> ProcessTableGuard<'_> {
        self.ports.guard()
    }

    /// Return a guard which can be used to access the registered names table safely
    ///
    /// See the note on `process_table_guard`, for notes on using guards in general.
    #[inline]
    pub fn name_table_guard(&self) -> ProcessTableGuard<'_> {
        self.names.guard()
    }

    /// Fetches the registrant associated with a registered name.
    ///
    /// If the name is not registered, or the registrant is dead, returns `None`
    pub fn get_by_name(&self, name: Atom, guard: &NameTableGuard<'_>) -> Option<Registrant> {
        self.names.get(&name, guard).and_then(|e| e.upgrade())
    }

    /// Fetches the process associated with the given process identifier.
    ///
    /// If the given pid is not associated with a registered process, returns `None`
    pub fn get_by_process_id(
        &self,
        pid: ProcessId,
        guard: &ProcessTableGuard<'_>,
    ) -> Option<Arc<Process>> {
        self.processes.get(&pid, guard).map(|p| p.clone())
    }

    /// Fetches the port associated with the given port identifier.
    ///
    /// If the given port id is not associated with a registered port, returns `None`
    pub fn get_by_port_id(&self, id: PortId, guard: &PortTableGuard<'_>) -> Option<Arc<Port>> {
        self.ports.get(&id, guard).map(|p| p.clone())
    }

    /// Registers a process by its process identifier, in the process table.
    ///
    /// A process identifier is only considered valid when it is associated with a process in the process table,
    /// so before a pid is used, or a process scheduled, it must have been registered.
    pub fn register_process(&self, process: Arc<Process>, guard: &ProcessTableGuard<'_>) {
        if let Err(err) = self.processes.try_insert(process.id(), process, guard) {
            panic!(
                "attempted to register a pid already in use {}",
                err.current.id()
            );
        }
    }

    /// Unregisters a process, given its process identifier, removing it from the process table.
    ///
    /// A process must be unregistered when it exits, to ensure that references to its process identifier
    /// are invalid.
    ///
    /// This function also unregisters any registered name associated with the given process before returning.
    pub fn unregister_process(
        &self,
        pid: ProcessId,
        guard: &ProcessTableGuard<'_>,
    ) -> Option<Arc<Process>> {
        let process = self.processes.remove(&pid, guard)?;
        let ntg = self.names.guard();
        self.unregister_name(Registrant::Process(process.clone()), &ntg);
        Some(process.clone())
    }

    /// Registers a port by its port identifier, in the port table.
    ///
    /// A port identifier is only considered valid when it is associated with a port in the port table,
    /// so before a port id is used, or a port scheduled, it must have been registered.
    pub fn register_port(&self, port: Arc<Port>, guard: &PortTableGuard<'_>) {
        if let Err(err) = self.ports.try_insert(port.id(), port, guard) {
            panic!(
                "attempted to register a port id already in use {}",
                err.current.id()
            );
        }
    }

    /// Unregisters a port, given its port identifier, removing it from the port table.
    ///
    /// A port must be unregistered when it exits, to ensure that references to its port identifier
    /// are invalid.
    ///
    /// This function also unregisters any registered name associated with the given port before returning.
    pub fn unregister_port(&self, id: PortId, guard: &PortTableGuard<'_>) -> Option<Arc<Port>> {
        let port = self.ports.remove(&id, guard)?;
        let ntg = self.names.guard();
        self.unregister_name(Registrant::Port(port.clone()), &ntg);
        Some(port.clone())
    }

    /// Registers `name` to the given registrant in the registered names table.
    ///
    /// This returns `Err` if the given name is already registered, or the process already has a registered name.
    pub fn register_name(
        &self,
        name: Atom,
        to: Registrant,
        guard: &NameTableGuard<'_>,
    ) -> Result<(), RegistrationError> {
        if unlikely(to.register_name(name).is_err()) {
            return Err(RegistrationError::AlreadyRegistered);
        }

        let weak = to.downgrade();

        match self.names.try_insert(name, weak, guard) {
            Ok(_) => Ok(()),
            Err(err) => {
                // Check to see if we can change ownership of the existing registration
                let mut success = false;
                self.names.compute_if_present(
                    &name,
                    |_k, v| {
                        match v.upgrade() {
                            None => {
                                // Name was previously registered, but process has since died;
                                // change ownership of the registration to `to`.
                                success = true;
                                Some(err.not_inserted)
                            }
                            Some(_) => {
                                // Name is registered by a process which is still alive;
                                // keep existing ownership.
                                Some(v.clone())
                            }
                        }
                    },
                    guard,
                );

                // If that failed, we have to return an error
                if success {
                    Ok(())
                } else {
                    to.unregister_name().unwrap();
                    Err(RegistrationError::AlreadyRegistered)
                }
            }
        }
    }

    /// Unregisters the name registered to `registrant` in the registered names table.
    pub fn unregister_name(&self, registrant: Registrant, guard: &NameTableGuard<'_>) {
        let Some(name) = registrant.registered_name() else { return; };
        if let Some(weak) = self.names.remove(&name, guard) {
            // This should never fail because `registrant` and `registered` should be the same process.
            //
            // If this does fail, it means that somehow another process/port was allowed to register a name which
            // ostensibly belongs to `registrant`, or vice/versa. In any case, it indicates a violation of what
            // we expect, so we want to panic if it happens.
            assert!(registrant.eq(weak));
            // Remove the name from the registrant itself
            registrant.unregister_name().ok();
        }
    }

    /// Returns an iterator over the registered names in this registry
    ///
    /// This iterator does not lock the registry, but it does prevent collection of garbage generated
    /// by modifications to the table while the iterator is alive.
    ///
    /// NOTE: It is not guaranteed that the iterator will see names which are registered after the iterator
    /// is returned but before it is finished iterating the registry. There is no defined order to the
    /// traversal either.
    pub fn names<'g>(
        &'g self,
        guard: &'g NameTableGuard<'_>,
    ) -> impl Iterator<Item = (Atom, WeakRegistrant)> + 'g {
        self.names.iter(guard).map(|(k, v)| (*k, v.clone()))
    }

    /// Returns the number of registered names in the registry
    pub fn registered_names<'g>(&'g self, _guard: &'g NameTableGuard<'_>) -> usize {
        self.names.len()
    }
}
