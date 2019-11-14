use crate::erts::term::prelude::*;
use crate::erts::testing::RegionHeap;

use super::*;

#[test]
fn sweep_tuple() {
    let mut fromspace = RegionHeap::new(default_heap_layout());
    let young = RegionHeap::new(default_heap_layout());
    let old = RegionHeap::new(default_heap_layout());
    let mut tospace = SemispaceHeap::new(young, old);
    // Allocate term in fromspace
    let tuple = fromspace
        .tuple_from_slice(&[atom!("hello"), atom!("world")])
        .unwrap();
    let tuple_ref = tuple.as_ref();
    // Sanity check
    assert_eq!(tuple_ref.len(), 2);

    // Get raw Term pointer
    let tuple_ptr: *mut Term = tuple.as_ptr() as *mut Term;

    // Sweep tuple into new young heap
    let mut sweeper = MinorCollection::new(&mut fromspace, &mut tospace);
    let result = unsafe { sweeper.sweep(tuple_ptr) };
    assert!(result.is_some());

    // We should have a new pointer
    let (new_ptr, bytes_moved) = result.unwrap();
    assert_ne!(tuple_ptr, new_ptr);
    // Should have moved term header + two atoms
    assert_eq!(bytes_moved, mem::size_of::<Term>() * 3);

    // Make sure moved term is still consistent
    let new_tuple = unsafe { Tuple::from_raw_term(new_ptr) };
    let new_tuple_ref = new_tuple.as_ref();
    assert_eq!(new_tuple_ref.len(), 2);
    assert_eq!(new_tuple_ref.get_element(0), Ok(atom!("hello")));
    assert_eq!(new_tuple_ref.get_element(1), Ok(atom!("world")));
}

#[test]
fn sweep_heapbin() {
    use crate::erts::string::Encoding;
   
    let mut fromspace = RegionHeap::new(default_heap_layout());
    let young = RegionHeap::new(default_heap_layout());
    let old = RegionHeap::new(default_heap_layout());
    let mut tospace = SemispaceHeap::new(young, old);
    // Allocate term in fromspace
    let heapbin = fromspace
        .heapbin_from_str("hello world!")
        .unwrap();
    let heapbin_ref = heapbin.as_ref();
    let heapbin_size = mem::size_of_val(heapbin_ref);
    // Sanity check
    assert_eq!(heapbin_ref.as_bytes(), "hello world!".as_bytes());

    // Sweep into new young heap
    let mut sweeper = MinorCollection::new(&mut fromspace, &mut tospace);
    let result = unsafe { sweeper.sweep(heapbin) };
    assert!(result.is_some());

    // We should have a new pointer
    let (new_ptr, bytes_moved) = result.unwrap();
    assert_ne!(heapbin.cast::<Term>().as_ptr(), new_ptr);
    // Should have moved heapbin header + "hello world!" bytes
    assert_eq!(bytes_moved, heapbin_size);

    // Make sure moved term is still consistent
    let new_heapbin = unsafe { HeapBin::from_raw_term(new_ptr) };
    let new_heapbin_ref = new_heapbin.as_ref();
    assert_eq!(new_heapbin_ref.as_bytes(), "hello world!".as_bytes());
    assert_eq!(new_heapbin_ref.encoding(), Encoding::Latin1);
}


#[test]
fn sweep_procbin() {
    use crate::erts::string::Encoding;

    let mut fromspace = RegionHeap::new(default_heap_layout());
    let young = RegionHeap::new(default_heap_layout());
    let old = RegionHeap::new(default_heap_layout());
    let mut tospace = SemispaceHeap::new(young, old);
    // Allocate term in fromspace
    let bin = fromspace
        .procbin_from_str("hello world!")
        .unwrap();
    let bin_ref = bin.as_ref();
    let bin_size = mem::size_of_val(bin_ref);
    // Sanity check
    assert_eq!(bin_ref.as_bytes(), "hello world!".as_bytes());

    // Sweep into new young heap
    let mut sweeper = MinorCollection::new(&mut fromspace, &mut tospace);
    let result = unsafe { sweeper.sweep(bin) };
    assert!(result.is_some());

    // We should have a new pointer
    let (new_ptr, bytes_moved) = result.unwrap();
    assert_ne!(bin.cast::<Term>().as_ptr(), new_ptr);
    // Should have moved procbin header
    assert_eq!(bytes_moved, bin_size);

    // Make sure moved term is still consistent
    let new_bin = Boxed::new(new_ptr as *mut ProcBin).unwrap();
    let new_bin_ref = new_bin.as_ref();
    assert_eq!(new_bin_ref.as_bytes(), "hello world!".as_bytes());
    assert_eq!(new_bin_ref.encoding(), Encoding::Latin1);
}

#[test]
fn sweep_procbin_matured() {
    use liblumen_core::util::pointer::in_area;
    use crate::erts::string::Encoding;

    let mut fromspace = RegionHeap::new(default_heap_layout());
    let young = RegionHeap::new(default_heap_layout());
    let old = RegionHeap::new(default_heap_layout());
    let mut tospace = SemispaceHeap::new(young, old);
    // Allocate term in fromspace
    let bin = fromspace
        .procbin_from_str("hello world!")
        .unwrap();
    let bin_ref = bin.as_ref();
    let bin_size = mem::size_of_val(bin_ref);
    // Sanity check
    assert_eq!(bin_ref.as_bytes(), "hello world!".as_bytes());
    // Set high water mark to top of heap
    fromspace.set_high_water_mark();
    assert_ne!(fromspace.heap_start(), fromspace.high_water_mark());
    // Make sure allocation is in mature region
    assert!(in_area(bin.as_ptr(), fromspace.heap_start(), fromspace.high_water_mark()));

    // Sweep into new young heap
    let mut sweeper = MinorCollection::new(&mut fromspace, &mut tospace);
    let result = unsafe { sweeper.sweep(bin) };
    assert!(result.is_some());

    // We should have a new pointer
    let (new_ptr, bytes_moved) = result.unwrap();
    assert_ne!(bin.cast::<Term>().as_ptr(), new_ptr);
    // Should have moved procbin header
    assert_eq!(bytes_moved, bin_size);
    // Make sure we're in old generation
    assert!(tospace.contains(new_ptr));
    assert!(!tospace.young_generation().contains(new_ptr));
    assert!(tospace.old_generation().contains(new_ptr));

    // Make sure moved term is still consistent
    let new_bin = Boxed::new(new_ptr as *mut ProcBin).unwrap();
    let new_bin_ref = new_bin.as_ref();
    assert_eq!(new_bin_ref.as_bytes(), "hello world!".as_bytes());
    assert_eq!(new_bin_ref.encoding(), Encoding::Latin1);
}
