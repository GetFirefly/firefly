use core::mem;
use core::ptr;

use crate::erts::*;

use super::alloc;

// This test ensures that after a full collection, expected garbage was cleaned up
#[test]
fn gc_simple_fullsweep_test() {
    // Create process
    let (heap, heap_size) = alloc::default_heap().unwrap();
    let process = ProcessControlBlock::new(heap, heap_size);
    process.set_flags(ProcessFlag::NeedFullSweep);
    simple_gc_test(process);
}

// This test ensures that after a minor collection, expected garbage was cleaned up
#[test]
fn gc_simple_minor_test() {
    // Create process
    let (heap, heap_size) = alloc::default_heap().unwrap();
    let process = ProcessControlBlock::new(heap, heap_size);
    simple_gc_test(process);
}

// This test is like `gc_simple_minor_test`, but also validates that after two collections,
// that objects which have survived (tenured objects), have been moved to the old
// generation heap
#[test]
fn gc_minor_tenuring_test() {
    unimplemented!()
}

// This test is a further extension of `gc_minor_tenuring_test` that ensures that a full sweep
// which occurs after tenuring has occured, results in all tenured objects being moved to a fresh
// young generation heap, with the old generation heap having been freed
#[test]
fn gc_fullsweep_after_tenuring_test() {
    unimplemented!()
}

fn simple_gc_test(mut process: ProcessControlBlock) {
    // Allocate an `{:ok, "hello world"}` tuple
    // First, the `ok` atom, an immediate, is super easy
    let ok = unsafe { Atom::try_from_str("ok").unwrap().as_term() };
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_term = make_binary_from_str(&mut process, "hello world").unwrap();
    // Finally, allocate room for the tuple itself, which is essentially an
    // array of `Term`, which in the case of immediates actually _is_ an array,
    // but as in our test here, when boxed terms are involved, doesn't fully
    // contain everything.
    let elements = [ok, greeting_term];
    let tuple_term = make_tuple_from_slice(&mut process, &elements).unwrap();
    assert!(tuple_term.is_boxed());
    let tuple_ptr = tuple_term.boxed_val();

    // Now, we will simulate updating the greeting of the above tuple with a new one,
    // leaving the original greeting dead, and a target for collection
    let new_greeting_term = make_binary_from_str(&mut process, "goodbye!").unwrap();

    // Update second element of the tuple above
    unsafe {
        let tuple_unwrapped = *tuple_ptr;
        assert!(tuple_unwrapped.is_tuple());
        let tuple = &*(tuple_ptr as *mut Tuple);
        let head_ptr = tuple.head();
        let second_element_ptr = head_ptr.offset(1);
        ptr::write(second_element_ptr, new_greeting_term);
    }

    // Grab current heap size
    let peak_size = process.young.heap_used();
    // Run garbage collection, using a pointer to the boxed tuple as our sole root
    let roots = [tuple_term];
    process.garbage_collect(0, &roots).unwrap();
    // Grab post-collection size
    let collected_size = process.young.heap_used();
    // We should be missing _exactly_ `greeting` bytes (rounded up to nearest word)
    let reclaimed = peak_size - collected_size;
    let greeting_size = greeting.len() + mem::size_of::<HeapBin>();
    let expected = to_word_size(greeting_size);
    assert_eq!(
        expected, reclaimed,
        "expected to reclaim {} words of memory, but reclaimed {} words",
        expected, reclaimed
    );
    // Assert that root has been updated, as the underlying object should have moved to a new heap
    let root = roots[0];
    assert!(root.is_boxed());
    //assert!(is_move_marker(unsafe { *root.boxed_val() }));
    let new_tuple_ptr = follow_moved(root).boxed_val();
    assert_ne!(new_tuple_ptr, tuple_ptr as *mut Term);
    // Assert that we can still access data that should be live
    let new_tuple = unsafe { &*(new_tuple_ptr as *mut Tuple) };
    assert_eq!(new_tuple.size(), 2);
    // First, the atom
    assert_eq!(Ok(ok), new_tuple.get_element_internal(1));
    // Then to validate the greeting, we need to follow the boxed term, unwrap it, and validate it
    let greeting_element = new_tuple.get_element_internal(2);
    assert!(greeting_element.is_ok());
    let greeting_box = greeting_element.unwrap();
    assert!(greeting_box.is_boxed());
    let greeting_ptr = greeting_box.boxed_val();
    let greeting_term = unsafe { *greeting_ptr };
    assert!(greeting_term.is_heapbin());
    let greeting_str = unsafe { &*(greeting_ptr as *mut HeapBin) };
    assert_eq!("goodbye!", greeting_str.as_str())
}
