use alloc::sync::Arc;
use core::any::TypeId;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};

use super::{Node, Term};

/// This struct abstracts over the locality of a port identifier.
#[derive(Debug, Clone)]
#[repr(u8)]
pub enum Port {
    Local {
        id: PortId,
    },
    External {
        id: PortId,
        node: Arc<Node>,
        next: *mut u8,
    },
}
impl Port {
    pub const TYPE_ID: TypeId = TypeId::of::<Port>();
}
impl TryFrom<Term> for Port {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Port(port) => Ok(Port::clone(port.as_ref())),
            _ => Err(()),
        }
    }
}
impl Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Local { id } => write!(f, "#Port<0.{}>", id.as_u64()),
            Self::External { id, node, .. } => write!(f, "#Port<{}.{}>", node.id(), id.as_u64()),
        }
    }
}
impl Eq for Port {}
impl PartialEq for Port {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Local { id: x }, Self::Local { id: y }) => x.eq(y),
            (
                Self::External {
                    id: xid,
                    node: xnode,
                    ..
                },
                Self::External {
                    id: yid,
                    node: ynode,
                    ..
                },
            ) => xnode.eq(ynode) && xid.eq(yid),
            _ => false,
        }
    }
}
impl Ord for Port {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (self, other) {
            (Self::Local { id: x }, Self::Local { id: y }) => x.cmp(y),
            (Self::Local { .. }, Self::External { .. }) => Ordering::Less,
            (
                Self::External {
                    id: xid,
                    node: xnode,
                    ..
                },
                Self::External {
                    id: yid,
                    node: ynode,
                    ..
                },
            ) => match xnode.cmp(ynode) {
                Ordering::Equal => xid.cmp(yid),
                other => other,
            },
            (Self::External { .. }, Self::Local { .. }) => Ordering::Greater,
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Local { id } => {
                id.hash(state);
            }
            Self::External { id, node, .. } => {
                id.hash(state);
                node.hash(state);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PortId(u64);
impl PortId {
    #[inline(always)]
    pub unsafe fn from_raw(id: u64) -> Self {
        Self(id)
    }

    #[inline(always)]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}
