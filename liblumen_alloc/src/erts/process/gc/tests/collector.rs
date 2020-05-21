use core::alloc::Layout;
use core::convert::TryInto;
use core::ffi::c_void;
use core::mem;
use core::ops::Deref;

use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::test::process;
use crate::erts::term::prelude::*;
use crate::erts::term::closure::*;
use crate::erts::*;
use crate::{atom, fixnum};

// This test ensures that after a full collection, expected garbage was cleaned up
#[test]
fn gc_simple_fullsweep_test() {
    // Create process
    let process = process();
    process.set_flags(ProcessFlags::NeedFullSweep);
    simple_gc_test(process);
}

// This test ensures that after a minor collection, expected garbage was cleaned up
#[test]
fn gc_simple_minor_test() {
    // Create process
    let process = process();
    simple_gc_test(process);
}

// This test is like `gc_simple_minor_test`, but also validates that after two collections,
// that objects which have survived (tenured objects), have been moved to the old
// generation heap
#[test]
#[ignore]
fn gc_minor_tenuring_test() {
    let process = process();
    tenuring_gc_test(process, false);
}

// This test is a further extension of `gc_minor_tenuring_test` that ensures that a full sweep
// which occurs after tenuring has occured, results in all tenured objects being moved to a fresh
// young generation heap, with the old generation heap having been freed
#[test]
#[ignore]
fn gc_fullsweep_after_tenuring_test() {
    let process = process();
    tenuring_gc_test(process, true);
}

fn simple_gc_test(process: Process) {
    // Allocate an `{:ok, "hello world"}` tuple
    // First, the `ok` atom, an immediate, is super easy
    let ok = atom!("ok");
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_hb = process.acquire_heap().heapbin_from_str(greeting).unwrap();
    let greeting_term: Term = greeting_hb.encode().unwrap();
    let greeting_size = Layout::for_value(greeting_hb.as_ref()).size();
    // Finally, allocate room for the tuple itself, which is essentially an
    // array of `Term`, which in the case of immediates actually _is_ an array,
    // but as in our test here, when boxed terms are involved, doesn't fully
    // contain everything.
    let elements = [ok, greeting_term];
    let tuple_term = process.tuple_from_slice(&elements).unwrap();
    assert!(tuple_term.is_boxed());
    let tuple_ptr: *mut Term = tuple_term.dyn_cast();

    // Allocate the list `[101, "test"]`
    let num = process.integer(101usize).unwrap();
    let string = "test";
    let string_term = process.binary_from_str(string).unwrap();

    assert!(string_term.is_boxed());
    let string_term_ptr: *mut Term = string_term.dyn_cast();
    let string_term_unboxed = unsafe { *string_term_ptr };
    assert!(string_term_unboxed.is_heapbin());
    let string_term_heap_bin = unsafe { HeapBin::from_raw_term(string_term_ptr) };
    assert_eq!(string_term_heap_bin.full_byte_len(), 4);
    assert_eq!("test", string_term_heap_bin.as_str());

    let list_nn = ListBuilder::new(&mut process.acquire_heap())
        .push(num)
        .push(string_term)
        .finish()
        .unwrap();
    let list_term: Term = list_nn.into();
    assert!(list_term.is_list());
    assert!(list_term.is_non_empty_list());
    let list_ptr: *const Cons = list_term.dyn_cast();
    assert_eq!(list_ptr, list_nn.as_ptr());

    // Allocate a closure with an environment [999, "this is a binary"]
    let closure_num = fixnum!(999);
    let closure_string = "this is a binary";
    let closure_string_term = process.binary_from_str(closure_string).unwrap();

    let creator = Pid::new(1, 0).unwrap();
    let module = atom_from_str!("module");

    extern "C" fn native() -> Term {
        Term::NONE
    }

    let index = 1 as Index;
    let old_unique = 2 as OldUnique;
    let unique = [0u8; 16];
    let closure = process
        .acquire_heap()
        .anonymous_closure_with_env_from_slices(
            module,
            index,
            old_unique,
            unique,
            2,
            Some(native as *const c_void),
            Creator::Local(creator),
            &[&[closure_num, closure_string_term]],
        )
        .unwrap();
    assert_eq!(closure.env_len(), 2);
    let closure_term: Term = closure.into();
    assert!(closure_term.is_boxed());
    let closure_ptr: *mut Term = closure_term.dyn_cast();
    let closure_term_ref = unsafe { Closure::from_raw_term(closure_ptr) };
    assert_eq!(closure_term_ref.env_len(), 2);
    assert!(closure_term_ref.get_env_element(0).is_smallint());
    assert!(closure_term_ref.get_env_element(1).is_boxed());

    // Now, we will simulate updating the greeting of the above tuple with a new one,
    // leaving the original greeting dead, and a target for collection
    let new_greeting_term = process.binary_from_str("goodbye!").unwrap();

    // Update second element of the tuple above
    let tuple_unwrapped = unsafe { &*tuple_ptr };
    assert!(tuple_unwrapped.is_tuple());
    let mut tuple = unsafe { Tuple::from_raw_term(tuple_ptr) };
    assert!(tuple.set_element(1, new_greeting_term).is_ok());

    // Grab current heap size
    let peak_size = process.young_heap_used();
    assert_eq!(process.stack_used(), 0);
    //dbg!(process.acquire_heap().heap().young_generation());

    // Run garbage collection
    let mut roots = [tuple_term, list_term, closure_term];
    process.garbage_collect(0, &mut roots).unwrap();
    process.set_flags(ProcessFlags::NeedFullSweep);
    // Grab post-collection size
    let collected_size_first = process.young_heap_used();

    //dbg!(process.acquire_heap().heap().young_generation());
    // Run it again and make sure the heap size stays the same
    process.garbage_collect(0, &mut roots).unwrap();
    let collected_size_second = process.young_heap_used();
    assert_eq!(collected_size_first, collected_size_second);

    // We should be missing _exactly_ x `greeting` bytes (rounded up to nearest word),
    dbg!(peak_size, collected_size_first);
    let reclaimed = peak_size - collected_size_first;
    let expected = to_word_size(greeting_size);
    assert_eq!(
        expected, reclaimed,
        "expected to reclaim {} words of memory, but reclaimed {} words",
        expected, reclaimed
    );
    // Assert that roots have been updated, as the underlying object should have moved to a new heap
    // First the tuple
    let tuple_root = roots[0];
    assert!(tuple_root.is_boxed());
    let new_tuple_ptr: *mut Term = tuple_root.follow_moved().dyn_cast();
    assert_ne!(new_tuple_ptr, tuple_ptr as *mut Term);
    // Assert that we can still access data that should be live
    let new_tuple = unsafe { Tuple::from_raw_term(new_tuple_ptr) };
    assert_eq!(new_tuple.len(), 2);
    // First, the atom
    assert_eq!(Ok(ok), new_tuple.get_element(0));
    // Then to validate the greeting, we need to follow the boxed term, unwrap it, and validate it
    let greeting_element = new_tuple.get_element(1);
    assert!(greeting_element.is_ok());
    let greeting_box = greeting_element.unwrap();
    assert!(greeting_box.is_boxed());
    let greeting_ptr: *mut Term = greeting_box.dyn_cast();
    let greeting_term = unsafe { *greeting_ptr };
    assert!(greeting_term.is_heapbin());
    let greeting_str = unsafe { HeapBin::from_raw_term(greeting_ptr) };
    assert_eq!("goodbye!", greeting_str.as_str());

    let list_root = roots[1];
    assert!(list_root.is_non_empty_list());
    let new_list_ptr: *const Cons = list_root.follow_moved().dyn_cast();
    assert_ne!(new_list_ptr, list_ptr);
    // Assert that we can still access list elements
    let new_list = unsafe { &*new_list_ptr };
    // The first value should be the integer
    assert!(new_list.head.is_smallint());
    // The tail should be another cons cell
    assert!(new_list.tail.is_non_empty_list());
    let new_list_tail: Boxed<Cons> = new_list.tail.try_into().unwrap();
    // The last value should be a heapbin
    assert!(new_list_tail.head.is_boxed());
    let test_string_ptr: *mut Term = new_list_tail.head.dyn_cast();
    let test_string = unsafe { HeapBin::from_raw_term(test_string_ptr) };
    assert_eq!("test", test_string.as_str());

    let closure_root = roots[2];
    assert!(closure_root.is_boxed());
    let closure_root_ptr: *mut Term = closure_root.dyn_cast();
    let closure_root_ref = unsafe { Closure::from_raw_term(closure_root_ptr) };
    assert!(closure_root_ref.env_len() == 2);
    assert!(closure_root_ref.get_env_element(0).is_smallint());
    assert!(closure_root_ref.get_env_element(1).is_boxed());
}

fn tenuring_gc_test(process: Process, _perform_fullsweep: bool) {
    // Allocate an `{:ok, "hello world"}` tuple
    // First, the `ok` atom, an immediate, is super easy
    let ok = atom!("ok");
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_term = process.binary_from_str(greeting).unwrap();
    // Construct tuple containing the atom and string
    let elements = [ok, greeting_term];
    let tuple_term = process.tuple_from_slice(&elements).unwrap();
    // Verify that the resulting tuple is valid
    assert!(tuple_term.is_boxed());
    let tuple_ptr: *mut Term = tuple_term.dyn_cast();
    let mut tup = unsafe { Tuple::from_raw_term(tuple_ptr) };
    let t1 = tup.get_element(0).unwrap();
    dbg!(t1);
    assert!(t1.is_atom());
    let t2 = tup.get_element(1).unwrap();
    dbg!(t2);
    assert!(t2.is_boxed());
    let t2ptr: *mut Term = t2.dyn_cast();
    let t2val = unsafe { *t2ptr };
    dbg!(t2val);
    assert!(t2val.is_heapbin());
    let t2str = unsafe { HeapBin::from_raw_term(t2ptr) };
    assert_eq!(t2str.as_str(), greeting);
    let greeting_size = Layout::for_value(t2str.as_ref()).size();

    // Allocate a list `[101, "this is a list"]`
    let num = fixnum!(101);
    let string = "this is a list";
    let string_term = process.binary_from_str(string).unwrap();
    let list = ListBuilder::new(&mut process.acquire_heap())
        .push(num)
        .push(string_term)
        .finish()
        .unwrap();
    let list_term: Term = list.encode().unwrap();
    assert!(list_term.is_non_empty_list());
    // Verify the resulting list is valid
    let cons_ptr: *const Cons = list_term.dyn_cast();
    let cons = unsafe { &*cons_ptr };
    let mut cons_iter = cons.into_iter();
    let l1 = cons_iter.next().unwrap().unwrap();
    dbg!(l1);
    assert!(l1.is_smallint());
    let l2 = cons_iter.next().unwrap().unwrap();
    dbg!(l2);
    assert!(l2.is_boxed());
    let l2ptr: *mut Term = l2.dyn_cast();
    let l2val = unsafe { *l2ptr };
    dbg!(l2val);
    assert!(l2val.is_heapbin());
    let l2str = unsafe { HeapBin::from_raw_term(l2ptr) };
    assert_eq!(l2str.as_str(), string);

    // Put term references on the stack
    process.stack_push(tuple_term).unwrap();
    process.stack_push(list_term).unwrap();

    // Now, we will simulate updating the greeting of the above tuple with a new one,
    // leaving the original greeting dead, and a target for collection
    let new_greeting = "goodbye world!";
    let new_greeting_term = process.binary_from_str(new_greeting).unwrap();

    // Update second element of the tuple above
    tup.set_element(1, new_greeting_term).unwrap();
    let t1 = tup.get_element(0).unwrap();
    assert!(t1.is_smallint());
    let t2 = tup.get_element(1).unwrap();
    assert!(t2.is_boxed());
    let t2ptr: *mut Term = t2.dyn_cast();
    let t2val = unsafe { *t2ptr };
    assert!(t2val.is_heapbin());
    let t2str = unsafe { HeapBin::from_raw_term(t2ptr) };
    assert_eq!(t2str.as_str(), new_greeting);

    // Grab current heap size
    let peak_size = process.young_heap_used();
    // Run first garbage collection
    let mut roots = [];
    process.garbage_collect(0, &mut roots).unwrap();

    // Verify size of garbage collected meets expectation
    let collected_size = process.young_heap_used();
    // We should be missing _exactly_ `greeting` bytes (rounded up to nearest word)
    let reclaimed = peak_size - collected_size;
    let expected = to_word_size(greeting_size);
    assert_eq!(
        expected, reclaimed,
        "expected to reclaim {} words of memory, but reclaimed {} words",
        expected, reclaimed
    );

    // Verify roots, starting with the list since it is on top of the stack
    let list_term = process.stack_pop().unwrap();
    assert!(list_term.is_non_empty_list());
    let list_term_ptr: *const Cons = list_term.dyn_cast();
    let list = unsafe { &*list_term_ptr };
    assert!(!list.is_move_marker());
    let mut list_iter = list.into_iter();
    let l0 = list_iter.next().unwrap().unwrap();
    assert_eq!(l0, fixnum!(101));
    let l1 = list_iter.next().unwrap().unwrap();
    assert!(l1.is_boxed());
    let l1ptr: *mut Term = l1.dyn_cast();
    let l1bin = unsafe { *l1ptr };
    assert!(l1bin.is_heapbin());
    let l1str = unsafe { HeapBin::from_raw_term(l1ptr) };
    assert_eq!(l1str.as_str(), string);
    assert_eq!(list_iter.next(), None);

    let tuple_term = process.stack_pop().unwrap();
    let tuple_ptr_postgc: *mut Term = tuple_term.dyn_cast();
    assert_eq!(tuple_ptr_postgc, tuple_ptr);
    let tuple_boxed = unsafe { *tuple_ptr };
    assert!(tuple_boxed.is_boxed());
    let moved_tuple_ptr: *mut Term = tuple_boxed.dyn_cast();
    assert_ne!(moved_tuple_ptr, tuple_ptr);
    assert_ne!(moved_tuple_ptr, tuple_ptr_postgc);
    let moved_tuple = unsafe { *moved_tuple_ptr };
    let moved_tuple_hdr = unsafe { &*(moved_tuple_ptr as *const Header<Tuple>) };
    assert_eq!(moved_tuple_hdr.arity(), 2);
    let tup = unsafe { Tuple::from_raw_term(moved_tuple_ptr) };
    let t1 = tup.get_element(0).unwrap();
    assert!(t1.is_smallint());
    let t2 = tup.get_element(1).unwrap();
    assert!(t2.is_boxed());
    let t2ptr: *mut Term = t2.dyn_cast();
    let t2val = unsafe { *t2ptr };
    assert!(t2val.is_heapbin());
    let t2str = unsafe { HeapBin::from_raw_term(t2ptr) };
    assert_eq!(t2str.as_str(), new_greeting);

    // Push tuple back on stack, resolving the move marker
    process.stack_push(moved_tuple).unwrap();

    // Allocate a fresh list for the young generation which references the older list,
    // e.g. will be equivalent to `[202, 101, "this is a list"]
    let num2 = fixnum!(202);
    let second_list = ListBuilder::new(&mut process.acquire_heap())
        .push(num2)
        .push(list_term)
        .finish()
        .unwrap();
    let second_list_term: Term = second_list.encode().unwrap();
    let second_list_ptr: *const Cons = second_list_term.dyn_cast();
    assert!(list_term.is_non_empty_list());
    let second_list = unsafe { &*second_list_ptr };
    let mut list_iter = second_list.into_iter();
    let l0 = list_iter.next().unwrap().unwrap();
    assert_eq!(l0, fixnum!(202));
    let l1 = list_iter.next().unwrap().unwrap();
    assert_eq!(l1, fixnum!(101));
    let l2 = list_iter.next().unwrap().unwrap();
    assert!(l2.is_boxed());
    let l2ptr: *mut Term = l2.dyn_cast();
    let l2bin = unsafe { *l2ptr };
    assert!(l2bin.is_heapbin());
    let l1str = unsafe { HeapBin::from_raw_term(l2ptr) };
    assert_eq!(l1str.as_str(), string);
    assert_eq!(list_iter.next(), None);

    // Push reference to new list on stack
    process.stack_push(second_list_term).unwrap();

    // Run second garbage collection, which should tenure everything except the new term we just
    // allocated
    let second_peak_size = process.young_heap_used();
    let mut roots = [];
    process.garbage_collect(0, &mut roots).unwrap();

    // Verify no garbage was collected, we should have just tenured some data,
    // the only data on the young heap should be a single cons cell
    let heap = process.acquire_heap();
    dbg!(heap.deref());
    drop(heap);
    let second_collected_size = process.young_heap_used();
    let newly_allocated_size = to_word_size(mem::size_of::<Cons>());
    assert_eq!(second_collected_size, newly_allocated_size);
    assert_eq!(
        second_peak_size + newly_allocated_size,
        second_collected_size
    );

    // TODO

    // Verify that we now have an old generation and that it is of the expected size
    assert!(process.has_old_heap());
    assert_eq!(
        process.old_heap_used(),
        collected_size,
        "expected tenuring of older young generation heap"
    );

    // Verify roots
    let list_term = process.stack_pop().unwrap();
    assert!(list_term.is_non_empty_list());
    let list_term_ptr: *const Cons = list_term.dyn_cast();
    let list = unsafe { &*list_term_ptr };
    assert!(!list.is_move_marker());
    let mut list_iter = list.into_iter();
    let l0 = list_iter.next().unwrap().unwrap();
    assert!(l0.is_smallint());
    let l1 = list_iter.next().unwrap().unwrap();
    assert!(l1.is_smallint());
    let l2 = list_iter.next().unwrap().unwrap();
    assert!(l2.is_boxed());
    let l2_ptr: *const Term = l2.dyn_cast();
    let l2_bin = unsafe { *l2_ptr };
    assert!(l2_bin.is_heapbin());
    assert_eq!(list_iter.next(), None);

    let tuple_root = process.stack_pop().unwrap();
    verify_tuple_root(tuple_root, tuple_ptr)

    // Assert that roots have been updated properly
    /*
    let list_root = process.stack_pop().unwrap();
    assert!(list_root.is_list());
    let new_list_ptr = list_root.dyn_cast() as *const Cons;
    assert_ne!(new_list_ptr, second_list_ptr);
    // Assert that we can still access list elements
    let new_list = unsafe { &*new_list_ptr };
    assert!(!new_list.is_move_marker());
    // The first value should be the integer
    assert!(new_list.head.is_smallint());
    // The tail should be another cons cell
    assert!(new_list.tail.is_list());
    let new_list_tail = unsafe { &*new_list.tail.dyn_cast() as *const Cons };
    // The last value should be a heapbin
    assert!(new_list_tail.head.is_boxed());
    let test_string_term_ptr = new_list_tail.head.dyn_cast() as *const Term;
    let test_string_term = unsafe { *test_string_term_ptr };
    assert!(test_string_term.is_heapbin());
    let test_string = unsafe { &*(test_string_term_ptr as *mut HeapBin) };
    assert_eq!("test", test_string.as_str());
    */
}

fn verify_tuple_root(tuple_root: Term, tuple_ptr: *mut Term) {
    assert!(tuple_root.is_boxed());
    let new_tuple_ptr: *mut Term = tuple_root.dyn_cast();
    assert_ne!(new_tuple_ptr, tuple_ptr as *mut Term);
    let new_tuple_term = unsafe { *new_tuple_ptr };
    assert!(!new_tuple_term.is_boxed());
    // Assert that we can still access data that should be live
    let new_tuple = unsafe { Tuple::from_raw_term(new_tuple_ptr) };
    assert_eq!(new_tuple.len(), 2);
    // First, the atom
    let ok = atom!("ok");
    assert_eq!(Ok(ok), new_tuple.get_element(0));
    // Then to validate the greeting, we need to follow the boxed term, unwrap it, and validate it
    let greeting_element = new_tuple.get_element(1);
    assert!(greeting_element.is_ok());
    let greeting_box = greeting_element.unwrap();
    assert!(greeting_box.is_boxed());
    let greeting_ptr: *mut Term = greeting_box.dyn_cast();
    let greeting_term = unsafe { *greeting_ptr };
    assert!(greeting_term.is_heapbin());
    let greeting_str = unsafe { HeapBin::from_raw_term(greeting_ptr) };
    assert_eq!("goodbye!", greeting_str.as_str());
}
