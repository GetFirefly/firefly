#![feature(crate_visibility_modifier)]
///! This library provides the means to access the LLVM-generated stack maps
///! included in a binary when the use of LLVM statepoints or patchpoints are
///! present in the IR used to generate objects in that binary.
///!
///! The LLVM documentation provides more detail on what stack maps, statepoints,
///! and patchpoints are, and how they are used by a compiler. Here are some
///! helpful links:
///!
///! - [StackMaps](https://llvm.org/docs/StackMaps.html)
///! - [Statepoints](https://llvm.org/docs/Statepoints.html)
///! - [Garbage Collection](https://llvm.org/docs/GarbageCollection.html#stack-map)
///!
///! The code in this library is based on the helpful
///! [llvm-statepoint-utils](https://github.com/kavon/llvm-statepoint-utils)
///! library by Kavon Favardin. The algorithm is essentially the same, but the code
///! is substantially different to take advantage of Rust features that are not
///! present in C99. In addition, we are plannning to generate our stack map table
///! ahead-of-time, rather than building it at runtime; but at the moment this is
///! runtime oriented in order to test the fit into the overall GC scheme we're using.

#[cfg(target_pointer_size = "64")]
compile_error!("stackmaps are currently only supported on 64-bit platforms");

#[macro_use]
extern crate lazy_static;

mod internal;
#[cfg(test)]
mod tests;

use core::mem;
use core::slice;

use hashbrown::HashMap;

use self::internal::*;

pub use self::internal::FunctionInfo;

extern "C" {
    // The section name on macOS is `__llvm_stackmaps`, but
    // on Linux it is `.llvm_stackmaps`, however the segment
    // name is the same on both.
    #[link_name = "__LLVM_STACKMAPS"]
    static STACK_MAP_HEADER: StackMapHeader;
}

lazy_static! {
    static ref STACK_MAP: StackMap = StackMap::build();
}

pub type ReturnAddress = *const u8;

pub struct StackMap {
    version: u8,
    functions: &'static [FunctionInfo],
    constants: &'static [u64],
    frame_infos: HashMap<ReturnAddress, FrameInfo>,
}
unsafe impl Sync for StackMap {}
unsafe impl Send for StackMap {}
impl StackMap {
    #[inline]
    pub fn version(&self) -> u8 {
        self.version
    }

    #[inline]
    pub fn find_frame(&self, addr: ReturnAddress) -> Option<&FrameInfo> {
        self.frame_infos.get(&addr)
    }

    #[inline(always)]
    pub fn functions(&self) -> &'static [FunctionInfo] {
        self.functions
    }

    #[inline(always)]
    pub fn constants(&self) -> &'static [u64] {
        self.constants
    }

    /// Get the generated StackMap, constructing it if this is the first access
    #[inline]
    pub fn get() -> &'static Self {
        &*STACK_MAP
    }

    fn build() -> Self {
        // Obtain reference to header and validate it before proceeding
        let header = unsafe { &STACK_MAP_HEADER };
        assert_eq!(header.version, 3, "unsupported version of LLVM StackMaps");
        assert_eq!(header._reserved1, 0, "expected zero");
        unsafe {
            assert_eq!(header._reserved2, 0, "expected zero");
        }

        let num_functions = header.num_functions as usize;
        let base = header as *const _ as *const u8;
        let functions_ptr = unsafe { base.add(mem::size_of::<StackMapHeader>()) };
        let functions =
            unsafe { slice::from_raw_parts(functions_ptr as *const FunctionInfo, num_functions) };

        let num_constants = header.num_constants as usize;
        let constants_ptr =
            unsafe { functions_ptr.add(mem::size_of::<FunctionInfo>() * num_functions) };
        let constants =
            unsafe { slice::from_raw_parts(constants_ptr as *const u64, num_constants) };

        // We have to construct the map of return addresses to frame info, since it is
        // not in a easily searchable format by default.
        let mut frame_infos = HashMap::with_capacity(header.num_records as usize);

        // This pointer marks the current position in the set of call site headers,
        // which starts right after the constants initially
        let mut callsite_ptr = unsafe {
            constants_ptr.add(mem::size_of::<u64>() * num_constants) as *const CallSiteHeader
        };

        // For each function, iterate over its call sites and generate frame information
        // to be stored in our stack map structure
        for fun in functions.iter() {
            for callsite in fun.callsites(callsite_ptr) {
                // Construct the state of the stack frame for this call site
                let frame_info = Self::generate_frame_info(fun, callsite);
                frame_infos.insert(frame_info.return_address, frame_info);

                // Update the base pointer after each iteration, so that we
                // can correctly construct the callsite iterator for the next
                // function
                callsite_ptr = callsite as *const CallSiteHeader;
            }
        }

        Self {
            version: header.version,
            functions,
            constants,
            frame_infos,
        }
    }

    fn generate_frame_info(fun: &FunctionInfo, callsite: &CallSiteHeader) -> FrameInfo {
        let fun_address = fun.address as *const u8;
        let return_address = unsafe { fun_address.offset(callsite.code_offset as isize) };
        let frame_size = fun.stack_size;

        // Now we parse the location array according to the specific type
        // of locations that statepoints emit.
        //
        // See http://llvm.org/docs/Statepoints.html#stack-map-format

        let num_locations = callsite.num_locations as usize;
        let locations_ptr =
            unsafe { (callsite as *const CallSiteHeader).add(1) as *const ValueLocation };
        let locations = unsafe { slice::from_raw_parts(locations_ptr, num_locations) };

        // The first 2 locations are constants we dont care about,
        // but if asserts are on we check that they're constants.
        debug_assert_eq!(locations[0].kind, LocationKind::Constant);
        debug_assert_eq!(locations[1].kind, LocationKind::Constant);

        // The 3rd constant describes the number of "deopt" parameters
        // that we should skip over.
        assert_eq!(locations[3].kind, LocationKind::Constant);
        let num_deopt = locations[3].offset;
        assert!(
            num_deopt >= 0,
            "expected non-negative number of deopt parameters"
        );

        // The remaining locations describe pointer that the GC should track, and use a special
        // format:
        //
        //   "Each record consists of a pair of Locations. The second element in the record
        //    represents the pointer (or pointers) which need updated. The first element in the
        //    record provides a pointer to the base of the object with which the pointer(s) being
        //    relocated is associated. This information is required for handling generalized
        //    derived pointers since a pointer may be outside the bounds of the original
        //    allocation, but still needs to be relocated with the allocation."
        //
        // NOTE that we are currently ignoring the following part of the documentation because
        // it doesn't make sense... locations have no size field:
        //
        //   "The Locations within each record may [be] a multiple of pointer size. In the later
        //    case, the record must be interpreted as describing a sequence of pointers and their
        //    corresponding base pointers. If the Location is of size N x sizeof(pointer), then
        //    there will be N records of one pointer each contained within the Location. Both
        //    Locations in a pair can be assumed to be of the same size."
        let num_skipped = 3 + num_deopt as usize;
        let num_locations = num_locations - num_skipped;
        let num_slots = num_locations / 2;
        assert_eq!(
            num_locations % 2,
            0,
            "expected an even number of pointer locations"
        );

        let mut slots = Vec::with_capacity(num_slots);

        let mut locs = locations.iter().skip(num_skipped);
        loop {
            if let Some(base) = locs.next() {
                let derived = locs.next().unwrap();

                // All locations must be indirects in order for it to be in the frame
                if !(base.is_indirect() && derived.is_indirect()) {
                    continue;
                }

                if !base.is_base_pointer(derived) {
                    continue;
                }

                // It is a base pointer, aka base is equivalent to derived, save it
                slots.push(Slot {
                    kind: PointerKind::Base,
                    offset: base.convert_offset(frame_size),
                });
            } else {
                break;
            }
        }

        // Since derived pointers come after base pointers in the vec, we can store
        // the number of base pointer slots and provide a fast way to derive a slice
        // of either base or derived pointers when needed
        let base_slots = slots.len();

        // Repeat for derived pointers; we know all locations are indirects now
        let mut locs = locations.iter().skip(num_skipped);
        loop {
            if let Some(base) = locs.next() {
                let derived = locs.next().unwrap();

                // Skipped in the first pass
                if !base.is_indirect() {
                    continue;
                }

                // Already processed, or derived is not an indirect
                if !base.is_base_pointer(derived) {
                    continue;
                }

                // Find the index in our frame corresponding to the base pointer
                let mut base_index = None;
                for (index, slot) in slots.iter().enumerate() {
                    if slot.offset == base.offset {
                        base_index = Some(index);
                        break;
                    }
                }

                // Save the derived pointers info
                let base_index = base_index.expect("couldn't find base for derived pointer");
                slots.push(Slot {
                    kind: PointerKind::Derived(base_index as u32),
                    offset: derived.convert_offset(frame_size),
                });
            } else {
                break;
            }
        }

        // There is no liveout information emitted for statepoints,
        // and we place faith in the input on that being the case
        //
        // Reference for the above can be found
        // [here](https://llvm.org/docs/Statepoints.html#safepoint-semantics-verification)

        // Now we can also construct the actual FrameInfo
        FrameInfo {
            return_address,
            size_in_bytes: frame_size,
            num_base_slots: base_slots,
            slots,
        }
    }
}

/// A compact representation of key information about a frame.
///
/// ## Stack Layout
///
/// In the following diagram, the stack grows downwards (towards lower addresses)
///
///     ...snip...
///     ------------- <- base 2
///     frame 2's return address
///     ------------- <- start of frame 2 (computed with base1 + base1's frame size)
///     frame 1's contents
///     ------------- <- base 1, aka base for offsets into frame 1 (8 bytes above start of frame 1)
///     frame 1's return address
///     ------------- <- start of frame 1 (what you get immediately after a callq)
pub struct FrameInfo {
    pub return_address: ReturnAddress,
    pub size_in_bytes: usize,
    // All base pointers come before derived pointers in the following vector.
    // By storing the number of base pointer slots present in the vector, we can
    // easily derive slices of either base or derived pointers as needed.
    pub num_base_slots: usize,
    slots: Vec<Slot>,
}
impl FrameInfo {
    /// Return a slice of base pointers
    #[inline]
    pub fn iter_base(&self) -> &[Slot] {
        &self.slots[..self.num_base_slots]
    }

    /// Return a slice of derived pointers
    #[inline]
    pub fn iter_derived(&self) -> &[Slot] {
        &self.slots[self.num_base_slots..]
    }

    #[inline]
    pub fn slots(&self) -> &[Slot] {
        self.slots.as_slice()
    }

    #[inline]
    pub fn base_slot(&self, index: usize) -> Option<&Slot> {
        self.iter_base().get(index)
    }

    /// Return the base pointer for a derived pointer
    pub fn get_base(&self, derived: &Slot) -> Option<&Slot> {
        assert_ne!(derived.kind, PointerKind::Base);

        match derived {
            Slot {
                kind: PointerKind::Derived(index),
                ..
            } => {
                let index = *index as usize;
                if index < self.num_base_slots {
                    self.slots.get(index)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Return an iterator over the derived pointers for a given base pointer
    pub fn get_derived(&self, base: &Slot) -> impl Iterator<Item = &Slot> {
        assert_eq!(base.kind, PointerKind::Base);

        let index = self.iter_base().iter().enumerate().find_map(|(i, b)| {
            if b == base {
                Some(i as u32)
            } else {
                None
            }
        });
        self.iter_derived()
            .iter()
            .filter(move |d| d.is_derived_from(index))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    // A base pointer
    Base,
    // A derived pointer, contains the index of the corresponding
    // base pointer from which it was derived
    Derived(u32),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Slot {
    // A negative kind means this is a base pointer,
    // A non-negative kind means this is a derived pointer,
    // i.e. derived from the base pointer in slot number `kind`
    pub kind: PointerKind,
    // Offset relative to the base of a frame
    // See the diagram in the `FrameInfo` doc for the defintion of "base"
    pub offset: i32,
}
impl Slot {
    fn is_derived_from(&self, index: Option<u32>) -> bool {
        if let Some(i) = index {
            match self.kind {
                PointerKind::Derived(di) => di == i,
                _ => false,
            }
        } else {
            false
        }
    }
}
