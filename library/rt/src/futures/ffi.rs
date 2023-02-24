use alloc::boxed::Box;
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop};
use core::task::{Context, RawWaker, RawWakerVTable, Waker};

/// This trait extends `core::task::Context` with helpers related to our `FfiContext`.
pub trait ContextExt {
    fn with_ffi_context<T, F: FnOnce(&mut FfiContext) -> T>(&mut self, closure: F) -> T;
}
impl<'a> ContextExt for Context<'a> {
    fn with_ffi_context<T, F>(&mut self, closure: F) -> T
    where
        F: FnOnce(&mut FfiContext) -> T,
    {
        static C_WAKER_VTABLE_OWNED: FfiWakerVTable = {
            unsafe extern "C-unwind" fn clone(data: *const FfiWakerBase) -> *const FfiWakerBase {
                let data = data as *mut FfiWaker;
                let waker: Waker = (*(*data).waker.owned).clone();
                Box::into_raw(Box::new(FfiWaker {
                    base: FfiWakerBase {
                        vtable: &C_WAKER_VTABLE_OWNED,
                    },
                    waker: WakerUnion {
                        owned: ManuallyDrop::new(waker),
                    },
                }))
                .cast()
            }
            // In this case, we must own `data`. This can only happen when the `data` pointer is returned from `clone`.
            // Thus it is `Box<FfiWaker>`.
            unsafe extern "C-unwind" fn wake(data: *const FfiWakerBase) {
                let b = Box::from_raw(data as *mut FfiWaker);
                ManuallyDrop::into_inner(b.waker.owned).wake();
            }
            unsafe extern "C-unwind" fn wake_by_ref(data: *const FfiWakerBase) {
                let data = data as *mut FfiWaker;
                (*data).waker.owned.wake_by_ref();
            }
            // Same as `wake`.
            unsafe extern "C-unwind" fn drop(data: *const FfiWakerBase) {
                let mut b = Box::from_raw(data as *mut FfiWaker);
                ManuallyDrop::drop(&mut b.waker.owned);
                mem::drop(b);
            }
            FfiWakerVTable {
                clone,
                wake,
                wake_by_ref,
                drop,
            }
        };

        static C_WAKER_VTABLE_REF: FfiWakerVTable = {
            unsafe extern "C-unwind" fn clone(data: *const FfiWakerBase) -> *const FfiWakerBase {
                let data = data as *mut FfiWaker;
                let waker: Waker = (*(*data).waker.reference).clone();
                Box::into_raw(Box::new(FfiWaker {
                    base: FfiWakerBase {
                        vtable: &C_WAKER_VTABLE_OWNED,
                    },
                    waker: WakerUnion {
                        owned: ManuallyDrop::new(waker),
                    },
                }))
                .cast()
            }
            unsafe extern "C-unwind" fn wake_by_ref(data: *const FfiWakerBase) {
                let data = data as *mut FfiWaker;
                (*(*data).waker.reference).wake_by_ref();
            }
            unsafe extern "C-unwind" fn unreachable(_: *const FfiWakerBase) {
                unreachable!()
            }
            FfiWakerVTable {
                clone,
                wake: unreachable,
                wake_by_ref,
                drop: unreachable,
            }
        };

        let waker = FfiWaker {
            base: FfiWakerBase {
                vtable: &C_WAKER_VTABLE_REF,
            },
            waker: WakerUnion {
                reference: self.waker(),
            },
        };

        // SAFETY: The behavior of `waker` is sane since we forward them to another valid Waker.
        // That waker must be safe to use due to the contract of `RawWaker::new`.
        let mut ctx = unsafe { FfiContext::new(&waker) };

        closure(&mut ctx)
    }
}

#[repr(transparent)]
pub struct FfiContext<'a> {
    waker: *const FfiWakerBase,
    _marker: PhantomData<&'a FfiWakerBase>,
}
impl<'a> FfiContext<'a> {
    unsafe fn new(waker: &'a FfiWaker) -> Self {
        Self {
            waker: waker as *const FfiWaker as *const FfiWakerBase,
            _marker: PhantomData,
        }
    }

    /// Runs a closure with this context as a normal `std::task::Context`.
    pub fn with_context<T, F>(&mut self, closure: F) -> T
    where
        F: FnOnce(&mut Context) -> T,
    {
        static RUST_WAKER_VTABLE: RawWakerVTable = {
            unsafe fn clone(data: *const ()) -> RawWaker {
                let waker = data.cast::<FfiWaker>();
                let cloned = ((*(*waker).base.vtable).clone)(waker.cast());
                RawWaker::new(cloned.cast(), &RUST_WAKER_VTABLE)
            }
            unsafe fn wake(data: *const ()) {
                let waker = data.cast::<FfiWaker>();
                ((*(*waker).base.vtable).wake)(waker.cast());
            }
            unsafe fn wake_by_ref(data: *const ()) {
                let waker = data.cast::<FfiWaker>();
                ((*(*waker).base.vtable).wake_by_ref)(waker.cast());
            }
            unsafe fn drop(data: *const ()) {
                let waker = data.cast::<FfiWaker>();
                ((*(*waker).base.vtable).drop)(waker.cast());
            }
            RawWakerVTable::new(clone, wake, wake_by_ref, drop)
        };

        // SAFETY: `waker`'s vtable functions must have sane behaviors, this is the contract of `FfiContext::new`
        let waker = unsafe {
            // The waker reference is borrowed from external context. We must not call drop on it.
            ManuallyDrop::new(Waker::from_raw(RawWaker::new(
                self.waker.cast(),
                &RUST_WAKER_VTABLE,
            )))
        };
        let mut ctx = Context::from_waker(&*waker);

        closure(&mut ctx)
    }
}

#[repr(transparent)]
struct FfiWakerBase {
    vtable: *const FfiWakerVTable,
}

#[repr(C)]
struct FfiWaker {
    base: FfiWakerBase,
    waker: WakerUnion,
}

#[repr(C)]
union WakerUnion {
    reference: *const Waker,
    owned: ManuallyDrop<Waker>,
    unknown: (),
}

#[repr(C)]
struct FfiWakerVTable {
    clone: unsafe extern "C-unwind" fn(*const FfiWakerBase) -> *const FfiWakerBase,
    wake: unsafe extern "C-unwind" fn(*const FfiWakerBase),
    wake_by_ref: unsafe extern "C-unwind" fn(*const FfiWakerBase),
    drop: unsafe extern "C-unwind" fn(*const FfiWakerBase),
}
