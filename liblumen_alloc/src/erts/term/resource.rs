/// A resource is something from a BIF or NIF that needs to be memory managed, but cannot be
/// converted to a normal Term.
use core::alloc::Layout;
use core::any::{Any, TypeId};
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr::NonNull;
use core::sync::atomic::{self, AtomicUsize};

use liblumen_core::sys;

use crate::erts::exception::system::Alloc;
use crate::erts::process::alloc::heap_alloc::HeapAlloc;
use crate::CloneToProcess;

use super::prelude::{TypeError, TypedTerm, Term, Header, Boxed};

pub struct Resource {
    reference_count: AtomicUsize,
    value: Box<dyn Any>,
}

impl Resource {
    fn alloc(value: Box<dyn Any>) -> Result<NonNull<Self>, Alloc> {
        let layout = Layout::new::<Self>();

        unsafe {
            match sys::alloc::alloc(layout) {
                Ok(non_null_u8) => {
                    let resource_ptr = non_null_u8.as_ptr() as *mut Self;
                    resource_ptr.write(Self {
                        reference_count: Default::default(),
                        value,
                    });

                    let non_null_resource = NonNull::new_unchecked(resource_ptr);

                    Ok(non_null_resource)
                }
                Err(_) => Err(alloc!()),
            }
        }
    }

    pub fn value(&self) -> &dyn Any {
        self.value.as_ref()
    }
}

impl Debug for Resource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Resource")
            .field("reference_count", &self.reference_count)
            .field(
                "value",
                &format_args!("Any with {:?}", self.value.type_id()),
            )
            .finish()
    }
}

impl Display for Resource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A reference to `Resource`
#[repr(C)]
pub struct Reference {
    header: Header<Reference>,
    resource: NonNull<Resource>,
}

impl Reference {
    pub fn new(value: Box<dyn Any>) -> Result<Self, Alloc> {
        let resource = Resource::alloc(value)?;
        let reference = Self {
            header: Default::default(),
            resource,
        };

        unsafe {
            resource
                .as_ref()
                .reference_count
                .fetch_add(1, atomic::Ordering::SeqCst);
        }

        Ok(reference)
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.value().downcast_ref()
    }

    pub fn type_id(&self) -> TypeId {
        self.value().type_id()
    }

    pub fn value(&self) -> &dyn Any {
        self.resource().value()
    }

    // Private

    fn resource(&self) -> &Resource {
        unsafe { self.resource.as_ref() }
    }
}

impl Clone for Reference {
    fn clone(&self) -> Self {
        self.resource()
            .reference_count
            .fetch_add(1, atomic::Ordering::AcqRel);

        Self {
            header: self.header.clone(),
            resource: self.resource,
        }
    }
}

impl CloneToProcess for Reference {
    fn clone_to_heap<A>(&self, heap: &mut A) -> Result<Term, Alloc>
    where
        A: ?Sized + HeapAlloc,
    {
        let layout = Layout::new::<Self>();

        let ptr = unsafe {
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            ptr.write(Self {
                header: self.header.clone(),
                resource: self.resource,
            });

            ptr
        };

        Ok(ptr.into())
    }
}

impl Debug for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Reference")
            .field("header", &self.header)
            .field(
                "resource",
                &format_args!("{:p} => {:?}", self.resource, unsafe {
                    self.resource.as_ref()
                }),
            )
            .finish()
    }
}

impl Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.resource())
    }
}

impl Hash for Reference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource.hash(state);
    }
}

impl PartialEq for Reference {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
    }
}
impl<T> PartialEq<Boxed<T>> for Reference
where
    T: PartialEq<Reference>,
{
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl TryFrom<TypedTerm> for Boxed<Reference> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::ResourceReference(reference) => Ok(reference),
            _ => Err(TypeError),
        }
    }
}
