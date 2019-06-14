use core::ptr;
use core::mem;

use crate::erts::*;

use super::alloc;

#[test]
fn simple_gc_test() {
    // Create process
    let (heap, heap_size) = alloc::default_heap().unwrap();
    let mut process = ProcessControlBlock::new(heap, heap_size);

    // Allocate an `{:ok, "hello world"}` tuple 

    // First, the `ok` atom, an immediate, is super easy
    let ok = unsafe { Atom::try_from_str("ok").unwrap().as_term() };
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_layout = HeapBin::layout(greeting);
    let greeting_size = greeting_layout.size();
    let greeting_ptr = unsafe { process.alloc_layout(greeting_layout).unwrap().as_ptr() as *mut HeapBin };
    let bin_ptr = unsafe { greeting_ptr.offset(1) as *mut u8 };
    let bin = unsafe { HeapBin::from_raw_utf8_parts(bin_ptr, greeting_size) };
    unsafe {
        // Copy header
        ptr::write(greeting_ptr, bin);
        // Copy contents
        ptr::copy_nonoverlapping(greeting.as_ptr(), bin_ptr, greeting.len());
    }
    // Finally, allocate room for the tuple itself, which is essentially an
    // array of `Term`, which in the case of immediates actually _is_ an array,
    // but as in our test here, when boxed terms are involved, doesn't fully
    // contain everything.
    let tuple_layout = Tuple::layout(2);
    let tuple_ptr = unsafe { process.alloc_layout(tuple_layout).unwrap().as_ptr() as *mut Tuple };
    let head_ptr = unsafe { tuple_ptr.offset(1) as *mut Term };
    let tuple = Tuple::new(2);
    unsafe {
        // Write header
        ptr::write(tuple_ptr, tuple);
        // Write `:ok`
        ptr::write(head_ptr, ok);
        // Write pointer to "hello world"
        ptr::write(head_ptr.offset(1), Term::from_raw(greeting_ptr as usize | Term::FLAG_BOXED));
    }

    // Now, we will simulate updating the greeting of the above tuple with a new one,
    // leaving the original greeting dead, and a target for collection
    let new_greeting = "goodbye!";
    let new_greeting_layout = HeapBin::layout(new_greeting);
    let new_greeting_size = new_greeting_layout.size();
    let new_greeting_ptr = unsafe { process.alloc_layout(new_greeting_layout).unwrap().as_ptr() as *mut HeapBin };
    let new_bin_ptr = unsafe { new_greeting_ptr.offset(1) as *mut u8 };
    let new_bin = unsafe { HeapBin::from_raw_utf8_parts(new_bin_ptr, new_greeting_size) };
    unsafe {
        // Copy header
        ptr::write(new_greeting_ptr, new_bin);
        // Copy contents
        ptr::copy_nonoverlapping(new_greeting.as_ptr(), new_bin_ptr, new_greeting.len());
    }
    // Update second element of the tuple above
    unsafe {
        let second_element_ptr = head_ptr.offset(1);
        ptr::write(second_element_ptr, Term::from_raw(new_greeting_ptr as usize | Term::FLAG_BOXED));
    }

    // Grab current heap size
    let peak_size = process.young.heap_used();
    // Run garbage collection, using a pointer to the boxed tuple as our sole root
    let roots = unsafe { [Term::from_raw(tuple_ptr as usize | Term::FLAG_BOXED)] };
    process.garbage_collect(0, &roots).unwrap();
    // Grab post-collection size
    let collected_size = process.young.heap_used();
    // We should be missing _exactly_ `greeting_layout` bytes (rounded up to nearest word)
    let reclaimed = peak_size - collected_size;
    let mut expected = greeting_size / mem::size_of::<Term>();
    if greeting_size % mem::size_of::<Term>() != 0 {
        expected += 1;
    }
    assert_eq!(reclaimed, expected, "expected to reclaim {} words of memory, but only reclaimed {} words", expected, reclaimed);
}