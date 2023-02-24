use alloc::sync::Arc;
use core::intrinsics::unlikely;
use core::marker::PhantomData;

use crossbeam_skiplist::SkipMap;

use crate::process::{Process, ProcessId};
use crate::term::{Atom, Port, PortId};

use super::{Registrant, RegistrationError, WeakRegistrant};

#[repr(transparent)]
pub struct ProcessTableGuard<'a>(PhantomData<&'a SkipMap<ProcessId, Arc<Process>>>);

#[repr(transparent)]
pub struct PortTableGuard<'a>(PhantomData<&'a SkipMap<PortId, Arc<Port>>>);

#[repr(transparent)]
pub struct NameTableGuard<'a>(PhantomData<&'a SkipMap<Atom, WeakRegistrant>>);

#[derive(Default)]
pub struct Registry {
    processes: SkipMap<ProcessId, Arc<Process>>,
    ports: SkipMap<PortId, Arc<Port>>,
    names: SkipMap<Atom, WeakRegistrant>,
}
impl Registry {
    #[inline]
    pub fn process_table_guard(&self) -> ProcessTableGuard<'_> {
        ProcessTableGuard(PhantomData)
    }

    #[inline]
    pub fn port_table_guard(&self) -> PortTableGuard<'_> {
        PortTableGuard(PhantomData)
    }

    #[inline]
    pub fn name_table_guard(&self) -> NameTableGuard<'_> {
        NameTableGuard(PhantomData)
    }

    pub fn get_by_name(&self, name: Atom, _guard: &NameTableGuard<'_>) -> Option<Registrant> {
        self.names.get(&name).and_then(|e| e.value().upgrade())
    }

    pub fn get_by_process_id(
        &self,
        pid: ProcessId,
        _guard: &ProcessTableGuard<'_>,
    ) -> Option<Arc<Process>> {
        self.processes.get(&pid).map(|p| p.value().clone())
    }

    pub fn get_by_port_id(&self, id: PortId, _guard: &PortTableGuard<'_>) -> Option<Arc<Port>> {
        self.ports.get(&id).map(|p| p.value().clone())
    }

    pub fn register_process(&self, process: Arc<Process>, _guard: &ProcessTableGuard<'_>) {
        let pid = process.id();
        if unlikely(self.processes.contains_key(&pid)) {
            panic!("attempted to register pid that is already in use {}", &pid);
        }

        self.processes.get_or_insert(pid, process);
    }

    pub fn unregister_process(
        &self,
        pid: ProcessId,
        _guard: &ProcessTableGuard<'_>,
    ) -> Option<Arc<Process>> {
        if let Some(entry) = self.processes.remove(&pid) {
            let process = entry.value().clone();
            let ntg = self.name_table_guard();
            self.unregister_name(Registrant::Process(process.clone()), &ntg);
            Some(process)
        } else {
            None
        }
    }

    pub fn register_port(&self, port: Arc<Port>, _guard: &PortTableGuard<'_>) {
        let id = port.id();
        if unlikely(self.ports.contains_key(&id)) {
            panic!(
                "attempted to register port id that is already in use {}",
                &id
            );
        }

        self.ports.get_or_insert(id, port);
    }

    pub fn unregister_port(&self, id: PortId, _guard: &PortTableGuard<'_>) -> Option<Arc<Port>> {
        if let Some(entry) = self.ports.remove(&id) {
            let port = entry.value().clone();
            let ntg = self.name_table_guard();
            self.unregister_name(Registrant::Port(port.clone()), &ntg);
            Some(port)
        } else {
            None
        }
    }

    pub fn register_name(
        &self,
        name: Atom,
        to: Registrant,
        _guard: &NameTableGuard<'_>,
    ) -> Result<(), RegistrationError> {
        if unlikely(to.register_name(name).is_err()) {
            return Err(RegistrationError::AlreadyRegistered);
        }

        let weak = to.downgrade();
        let entry = self.names.get_or_insert(name, weak.clone());
        if entry.value() == &weak {
            Ok(())
        } else {
            Err(RegistrationError::AlreadyRegistered)
        }
    }

    pub fn unregister_name(&self, registrant: Registrant, _guard: &NameTableGuard<'_>) {
        let Some(name) = registrant.registered_name() else { return; };
        if let Some(entry) = self.names.remove(&name) {
            // This should never fail because `registrant` and `registered` should be the same process.
            //
            // If this does fail, it means that somehow another process/port was allowed to register a name which
            // ostensibly belongs to `registrant`, or vice/versa. In any case, it indicates a violation of what
            // we expect, so we want to panic if it happens.
            assert!(registrant.eq(entry.value()));
            // Remove the name from the registrant itself
            registrant.unregister_name().unwrap();
        }
    }

    pub fn names(
        &self,
        _guard: &NameTableGuard<'_>,
    ) -> impl Iterator<Item = (Atom, WeakRegistrant)> + '_ {
        self.names.iter().map(|e| (*e.key(), e.value().clone()))
    }

    /// Returns the number of registered names in the registry
    pub fn registered_names<'g>(&'g self, _guard: &'g NameTableGuard<'_>) -> usize {
        self.names.len()
    }
}
