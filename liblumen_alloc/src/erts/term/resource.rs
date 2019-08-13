/// A resource is something from a BIF or NIF that needs to be memory managed, but cannot be
/// converted to a normal Term.
use core::alloc::Layout;
use core::any::{Any, TypeId};
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr::NonNull;
use core::sync::atomic::{self, AtomicUsize};

use liblumen_core::sys;

use crate::erts::exception::system::Alloc;
use crate::erts::process::alloc::heap_alloc::HeapAlloc;
use crate::erts::term::term::Term;
use crate::erts::term::{arity_of, AsTerm, TypeError, TypedTerm};
use crate::CloneToProcess;

#[derive(Debug)]
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

impl Display for Resource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A reference to `Resource`
pub struct Reference {
    header: Term,
    resource: NonNull<Resource>,
}

impl Reference {
    pub unsafe fn from_raw(ptr: *mut Self) -> Self {
        let reference = &*ptr;

        reference.clone()
    }

    pub fn new(value: Box<dyn Any>) -> Result<Self, Alloc> {
        let resource = Resource::alloc(value)?;

        Ok(Self {
            header: Term::make_header(arity_of::<Self>(), Term::FLAG_RESOURCE_REFERENCE),
            resource,
        })
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

unsafe impl AsTerm for Reference {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}

impl Clone for Reference {
    fn clone(&self) -> Self {
        self.resource()
            .reference_count
            .fetch_add(1, atomic::Ordering::AcqRel);

        Self {
            header: self.header,
            resource: self.resource,
        }
    }
}

impl CloneToProcess for Reference {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        let layout = Layout::new::<Self>();

        let ptr = unsafe {
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            ptr.write(Self {
                header: self.header,
                resource: self.resource,
            });

            ptr
        };

        let boxed = Term::make_boxed(ptr);

        Ok(boxed)
    }
}

impl Debug for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Reference")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("resource", &self.resource)
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

impl TryFrom<Term> for Reference {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Reference {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::ResourceReference(reference) => Ok(reference),
            _ => Err(TypeError),
        }
    }
}
