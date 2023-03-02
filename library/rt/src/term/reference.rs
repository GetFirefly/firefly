use alloc::sync::Arc;
use core::any::Any;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::AtomicU64;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;
use firefly_system::sync::OnceLock;
use firefly_system::time::MonotonicTime;

use crate::gc::Gc;
use crate::scheduler::SchedulerId;
use crate::services::distribution::Node;

use super::{Boxable, Header, Pid, Tag};

/// This struct abstracts over the various types of reference payloads
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Reference {
    header: Header,
    id: ReferenceId,
    data: ReferenceType,
}

#[derive(Debug, Clone)]
pub enum ReferenceType {
    /// A simple reference, created locally
    Local,
    /// A local reference associated to a pid
    Pid(Pid),
    /// A reference containing an atomically reference-counted value of any type
    ///
    /// Magic values are used to represent various internal runtime structures in Erlang code
    /// as references. These can then be used to call a variety of special BIF/NIF functions for
    /// interacting with values of the underlying type.
    ///
    /// Magic values must be allocated via `Arc` and castable to `dyn Any + Send + Sync` for the
    /// following reasons:
    ///
    /// * We need the ability to represent any type via magic, safely
    /// * A raw pointer would prevent reference-counted values from being tracked properly
    /// * We can't guarantee that a reference won't be interacted with from multiple threads at the
    ///   same time, or sent between threads. So it is not safe to use a different container than
    ///   `Arc`, or omit the `Send` and `Sync` traits.
    ///
    /// If a value can't meet the above criteria, it can't be stored as magic directly, and you
    /// will likely need some intermediate type to use as the magic data.
    Magic(Arc<dyn Any + Send + Sync>),
    #[allow(unused)]
    External(Arc<Node>),
}

impl Boxable for Reference {
    type Metadata = ();

    const TAG: Tag = Tag::Reference;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Gc::new_uninit_in(heap).unwrap();
            unsafe {
                self.write_clone_into_raw(cloned.as_mut_ptr());
                cloned.assume_init()
            }
        }
    }
}

impl Reference {
    /// Make a new, unique reference not associated with any scheduler
    pub fn make() -> Self {
        Self {
            header: Header::new(Tag::Reference, 0),
            id: ReferenceId::next(),
            data: ReferenceType::Local,
        }
    }

    #[inline]
    pub fn new(id: ReferenceId) -> Self {
        Self {
            header: Header::new(Tag::Reference, 0),
            id,
            data: ReferenceType::Local,
        }
    }

    /// Create a new magic ref from the given reference id and raw pointer
    ///
    /// The pointer must be to a type that implements `Any`, so that it can be safely downcast
    /// to a type later on. It is best if magic references are "owned" by some type which
    /// understands what type the magic reference should be, as there could be any number of
    /// underlying types allocated in magic references throughout the system. For example, in
    /// the distribution system, connections are associated with magic references so that they
    /// can be returned as handles to Erlang code.
    ///
    /// NOTE: This function will panic if the given reference id is not a magic reference id
    pub fn new_magic(id: ReferenceId, ptr: Arc<dyn Any + Send + Sync>) -> Self {
        assert!(id.is_magic());
        Self {
            header: Header::new(Tag::Reference, 0),
            id,
            data: ReferenceType::Magic(ptr),
        }
    }

    /// Creates a new pid ref from the given reference id and pid
    ///
    /// NOTE: This function will panic if the given reference id is not a pid reference id
    pub fn new_pid(id: ReferenceId, pid: Pid) -> Self {
        assert!(id.is_pid());
        Self {
            header: Header::new(Tag::Reference, 0),
            id,
            data: ReferenceType::Pid(pid),
        }
    }

    /// Return the underlying reference identifier for this ref
    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.id
    }

    #[inline]
    pub fn is_local(&self) -> bool {
        !self.is_external()
    }

    #[inline]
    pub fn is_external(&self) -> bool {
        match &self.data {
            ReferenceType::External(_) => true,
            _ => false,
        }
    }

    /// If this is a magic reference, returns a new strong reference to the magic value
    pub fn magic(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        match &self.data {
            ReferenceType::Magic(ptr) => Some(ptr.clone()),
            _ => None,
        }
    }

    /// Returns the pid, if this is a pid reference
    pub fn pid(&self) -> Option<Pid> {
        match &self.data {
            ReferenceType::Pid(pid) => Some(pid.clone()),
            _ => None,
        }
    }

    /// Returns the node, if this is an external reference
    pub fn node(&self) -> Option<Arc<Node>> {
        match &self.data {
            ReferenceType::External(node) => Some(node.clone()),
            _ => None,
        }
    }
}
impl Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.data {
            ReferenceType::External(ref node) => write!(f, "#Ref<{}.{}>", node.id(), self.id),
            _ => write!(f, "#Ref<0.{}>", self.id),
        }
    }
}
impl Eq for Reference {}
impl crate::cmp::ExactEq for Reference {}
impl PartialEq for Reference {
    fn eq(&self, other: &Self) -> bool {
        if core::mem::discriminant(&self.data) != core::mem::discriminant(&other.data) {
            return false;
        }
        if self.id != other.id {
            return false;
        }
        match (&self.data, &other.data) {
            (ReferenceType::External(ref a), ReferenceType::External(ref b)) => a.eq(b),
            (ReferenceType::External(_), _) => false,
            (_, ReferenceType::External(_)) => false,
            _ => true,
        }
    }
}
impl PartialEq<Gc<Reference>> for Reference {
    fn eq(&self, other: &Gc<Reference>) -> bool {
        self.eq(other.deref())
    }
}
impl PartialOrd for Reference {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Reference {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (&self.data, &other.data) {
            (ReferenceType::External(ref a), ReferenceType::External(ref b)) => {
                a.cmp(b).then(self.id.cmp(&other.id))
            }
            (ReferenceType::External(_), _) => Ordering::Greater,
            (_, ReferenceType::External(_)) => Ordering::Less,
            _ => self.id.cmp(&other.id),
        }
    }
}
impl Hash for Reference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        core::mem::discriminant(&self.data).hash(state);
        match &self.data {
            ReferenceType::Local => (),
            ReferenceType::Pid(ref pid) => {
                pid.hash(state);
            }
            ReferenceType::Magic(ref ptr) => {
                ptr::hash(Arc::as_ptr(ptr), state);
            }
            ReferenceType::External(ref node) => {
                node.hash(state);
            }
        }
    }
}

const REF_NUMBERS: usize = 3;
const REF_NUM_SIZE: u32 = 18;
const MAGIC_MARKER_BIT_NO: u32 = REF_NUM_SIZE - 1;
const MAGIC_MARKER_BIT: u32 = 1 << MAGIC_MARKER_BIT_NO;
const PID_MARKER_BIT_NO: u32 = REF_NUM_SIZE - 2;
const PID_MARKER_BIT: u32 = 1 << PID_MARKER_BIT_NO;
const THR_ID_MASK: u32 = PID_MARKER_BIT - 1;
const NUM_MASK: u32 = !(THR_ID_MASK | MAGIC_MARKER_BIT | PID_MARKER_BIT);
const REF_MASK: u32 = !(u32::MAX << REF_NUM_SIZE);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId([u32; REF_NUMBERS]);
impl ReferenceId {
    /// Generates the next available global reference id
    ///
    /// You should prefer to request references from a scheduler, but in some cases that isn't
    /// possible
    pub fn next() -> Self {
        use core::sync::atomic::Ordering;
        static COUNTER: OnceLock<AtomicU64> = OnceLock::new();

        let counter = COUNTER.get_or_init(|| AtomicU64::new(ReferenceId::init()));

        let id = counter.fetch_add(1, Ordering::Acquire);

        unsafe { Self::new(SchedulerId::from_raw(0), id) }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self([0; REF_NUMBERS])
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0; REF_NUMBERS]
    }

    /// Create a `ReferenceId` from a given scheduler id and unique identifier
    ///
    /// # SAFETY
    ///
    /// Callers _must_ ensure that the `id` value they provide is unique for a given scheduler id.
    /// References are supposed to be globally unique, so failing to uphold this invariant will
    /// result in potentially duplicate references being generated which are supposed to mean
    /// different things.
    pub const unsafe fn new(scheduler_id: SchedulerId, id: u64) -> Self {
        // Don't use thread id in the first 18-bit word, since the hash/phash/phash2 bifs only hash
        // on that word, which would result in poor hash values. Instead, shuffle the bits a
        // bit.
        Self([
            (id & (REF_MASK as u64)) as u32,
            (id & (NUM_MASK as u64)) as u32 | (scheduler_id.as_u16() as u32 & THR_ID_MASK),
            ((id >> 32) & (u16::MAX as u64)) as u32,
        ])
    }

    /// Returns true if this reference id belongs to a magic reference
    pub fn is_magic(&self) -> bool {
        self.0[1] & MAGIC_MARKER_BIT == MAGIC_MARKER_BIT
    }

    /// Mark this reference id as belonging to a magic reference
    pub fn set_magic(&mut self) {
        self.0[1] |= MAGIC_MARKER_BIT;
    }

    /// Returns true if this reference id belongs to a pid reference
    pub fn is_pid(&self) -> bool {
        self.0[1] & PID_MARKER_BIT == PID_MARKER_BIT
    }

    /// Mark this reference id as belonging to a pid reference
    pub fn set_pid(&mut self) {
        self.0[1] |= PID_MARKER_BIT;
    }

    /// Returns the scheduler id which produced this reference id
    pub fn scheduler_id(&self) -> SchedulerId {
        unsafe { SchedulerId::from_raw((self.0[1] & THR_ID_MASK) as u16) }
    }

    /// Returns the reference number part of this reference id
    pub fn number(&self) -> u64 {
        let id_lo = self.0[0] as u64 | (self.0[1] & NUM_MASK) as u64;
        let id_hi = (self.0[2] as u64) << 32;
        id_hi | id_lo
    }

    /// Produces a pseudo-random initial reference id seed value based on the current monotonic
    /// time.
    ///
    /// This is based on the ERTS implementation in `erl_bif_unique.h`
    ///
    /// This is only called once when the system starts.
    pub fn init() -> u64 {
        let mut id = 0;
        let time = MonotonicTime::now();
        let duration = time.elapsed();
        let us = duration.subsec_micros() as u64;
        id |= duration.as_secs();
        id |= us << 32;
        id = id.wrapping_mul(268438039);
        id.wrapping_add(us)
    }
}
impl Display for ReferenceId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let r0 = self.0[0];
        let r1 = self.0[1];
        let r2 = self.0[2];
        write!(f, "{}.{}.{}", r0, r1, r2)
    }
}
