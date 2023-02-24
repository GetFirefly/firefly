use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::sync::atomic::Ordering;

use firefly_system::sync::Atomic;

use crate::drivers::{Driver, DriverError, LoadableDriver};
use crate::services::distribution::Node;

use super::{atoms, Atom, Header, Pid, Tag};

#[derive(Debug)]
pub struct Port {
    #[allow(unused)]
    header: Header,
    id: PortId,
    node: Option<Arc<Node>>,
    #[allow(unused)]
    owner: Pid,
    registered_name: Atomic<Atom>,
    #[allow(unused)]
    info: Option<PortInfo>,
}
impl Port {
    pub fn new(
        owner: Pid,
        command: &str,
        loadable_driver: &dyn LoadableDriver,
    ) -> Result<Arc<Self>, DriverError> {
        let mut handle = Arc::<Self>::new_uninit();
        let driver = loadable_driver.start(handle.clone(), command)?;

        unsafe {
            Arc::get_mut_unchecked(&mut handle).write(Self {
                header: Header::new(Tag::Port, 0),
                id: PortId::next(),
                node: None,
                owner,
                registered_name: Atomic::new(atoms::Undefined),
                info: Some(PortInfo {
                    name: command.to_string(),
                    driver,
                }),
            });
            Ok(handle.assume_init())
        }
    }

    #[cfg(test)]
    pub(crate) fn new_with_id(
        id: PortId,
        owner: Pid,
        command: &str,
        loadable_driver: &dyn LoadableDriver,
    ) -> Result<Arc<Port>, DriverError> {
        let mut handle = Arc::<Port>::new_uninit();
        let driver = loadable_driver.start(handle.clone(), command)?;

        unsafe {
            Arc::get_mut_unchecked(&mut handle).write(Self {
                header: Header::new(Tag::Port, 0),
                id,
                node: None,
                owner,
                registered_name: Atomic::new(atoms::Undefined),
                info: Some(PortInfo {
                    name: command.to_string(),
                    driver,
                }),
            });
            Ok(handle.assume_init())
        }
    }

    #[inline(always)]
    pub fn id(&self) -> PortId {
        self.id
    }

    #[inline]
    pub fn node(&self) -> Option<Arc<Node>> {
        self.node.clone()
    }

    #[inline]
    pub fn is_local(&self) -> bool {
        self.node.is_none()
    }

    #[inline]
    pub fn is_external(&self) -> bool {
        self.node.is_some()
    }

    pub fn registered_name(&self) -> Option<Atom> {
        let name = self.registered_name.load(Ordering::Relaxed);
        if name == atoms::Undefined {
            None
        } else {
            Some(name)
        }
    }

    /// Sets the registered name of this port
    ///
    /// This function returns `Ok` if the port was unregistered when this function was called,
    /// otherwise it returns `Err` with the previously registered name of this port. This function
    /// will never replace an already registered name.
    pub fn register_name(&self, name: Atom) -> Result<(), Atom> {
        assert_ne!(
            name,
            atoms::Undefined,
            "undefined is not a valid registered name"
        );
        match self.registered_name.compare_exchange(
            atoms::Undefined,
            name,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(existing) => {
                // Already registered
                Err(existing)
            }
        }
    }

    /// Removes the registered name of this port
    ///
    /// This function returns `Ok` if the port was registered when this function was called,
    /// otherwise it returns `Err` which implies that the port had no registered name already.
    pub fn unregister_name(&self) -> Result<(), ()> {
        let prev = self
            .registered_name
            .swap(atoms::Undefined, Ordering::Release);
        if prev == atoms::Undefined {
            Err(())
        } else {
            Ok(())
        }
    }
}
impl Eq for Port {}
impl crate::cmp::ExactEq for Port {}
impl PartialEq for Port {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) && self.node.eq(&other.node)
    }
}
impl PartialEq<Arc<Port>> for Port {
    fn eq(&self, other: &Arc<Port>) -> bool {
        self.eq(other.deref())
    }
}
impl Ord for Port {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (self.node.as_ref(), other.node.as_ref()) {
            (None, None) => self.id.cmp(&other.id),
            (None, Some(_)) => Ordering::Less,
            (Some(a), Some(b)) => a.cmp(&b).then_with(|| self.id.cmp(&other.id)),
            _ => Ordering::Greater,
        }
    }
}
impl PartialOrd for Port {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash for Port {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.node.hash(state);
    }
}
impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(node) = self.node.as_ref() {
            write!(f, "#Port<{}.{}>", node.id(), &self.id)
        } else {
            write!(f, "#Port<0.{}>", &self.id)
        }
    }
}

pub struct PortInfo {
    name: String,
    #[allow(unused)]
    driver: Box<dyn Driver>,
}
impl fmt::Debug for PortInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("PortInfo")
            .field("name", &self.name)
            .finish()
    }
}

/// Uniquely identifies an instance of a [`Port`] system-wide.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PortId(u64);
impl PortId {
    /// Generates the next port identifier for the local node
    pub fn next() -> Self {
        use core::sync::atomic::AtomicU64;

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    #[inline]
    pub const fn into_raw(&self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}
impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
