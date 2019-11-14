use crate::erts::term::prelude::*;
use crate::erts::testing::RegionHeap;

use super::*;

#[test]
fn simple_collector_test() {
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
    let mut tuple_root: Term = tuple_ptr.into();

    // Construct rootset pointing to our single root
    let mut roots = RootSet::new(&mut []);
    roots.push(&mut tuple_root);
    // Collect into new young heap using SimpleCollector
    let sweeper = MinorCollection::new(&mut fromspace, &mut tospace);
    let mut collector = SimpleCollector::new(roots, sweeper);
    let moved = collector.garbage_collect().unwrap();
    assert_eq!(moved, mem::size_of::<Term>() * 3);

    // We should have a move marker in `tuple_root`
    let new_tuple_ptr: *mut Term = tuple_root.dyn_cast();
    assert_ne!(tuple_ptr, new_tuple_ptr);
    assert!(unsafe { (*new_tuple_ptr).is_tuple() });
    // Follow marker and make sure tuple is good to go
    assert!(tospace.young_generation().contains(new_tuple_ptr));
    let new_tuple = unsafe { Tuple::from_raw_term(new_tuple_ptr) };
    let new_tuple_ref = new_tuple.as_ref();
    assert_eq!(new_tuple_ref.len(), 2);
    assert_eq!(new_tuple_ref.get_element(0), Ok(atom!("hello")));
    assert_eq!(new_tuple_ref.get_element(1), Ok(atom!("world")));
}
