#![no_std]
#![allow(incomplete_features)]
// Used for atom linkage
#![feature(linkage)]
// Used for syntax sugar
// Used for the `unlikely` compiler hint
#![feature(core_intrinsics)]
// Used for custom allocators
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(layout_for_ptr)]
#![feature(new_uninit)]
// Used with MaybeUninit
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
// Used in process heap impl
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
// Used for access to pointer metadata
#![feature(ptr_metadata)]
#![feature(ptr_mask)]
#![feature(ptr_sub_ptr)]
#![feature(pointer_byte_offsets)]
#![feature(pointer_is_aligned)]
// const BinaryData fns
#![feature(const_heap)]
#![feature(const_align_offset)]
#![feature(const_alloc_layout)]
#![feature(const_ptr_write)]
#![feature(const_format_args)]
#![feature(int_roundings)]
// Used for implementing Fn, etc. for Gc, as
// well as interop with closures produced by the compiler
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(tuple_trait)]
// Atom table
#![feature(return_position_impl_trait_in_trait)]
// Use of Gc<T> as receiver
#![feature(receiver_trait)]
// Used for const TypeId::of::<T>()
#![feature(const_type_id)]
// Used for NonNull::as_uninit_mut
#![feature(ptr_as_uninit)]
// Used for Arc::get_mut_unchecked
#![feature(get_mut_unchecked)]
// The following are used for the Tuple implementation
#![feature(trusted_len)]
#![feature(slice_index_methods)]
#![feature(slice_split_at_unchecked)]
#![feature(exact_size_is_empty)]
// Specialization
#![feature(min_specialization)]
// Used for FFI
#![feature(extern_types)]
#![feature(c_unwind)]
#![cfg_attr(test, feature(test))]
// Used for ErlangResult
#![feature(try_trait_v2)]
#![feature(try_trait_v2_residual)]
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
// Used for Map
#![feature(is_sorted)]
#![feature(iter_order_by)]
// Raw thread locals
#![feature(thread_local)]
// Testing
#![feature(assert_matches)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;
#[cfg(test)]
extern crate test;

pub mod backtrace;
pub mod bifs;
pub mod cmp;
pub mod drivers;
pub mod error;
pub mod function;
pub mod fundamental;
pub mod gc;
pub mod intrinsics;
pub mod process;
pub mod scheduler;
pub mod services;
pub mod term;
