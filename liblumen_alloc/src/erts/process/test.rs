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
#[ignore]
fn gc_minor_tenuring_test() {
    let (heap, heap_size) = alloc::default_heap().unwrap();
    let process = ProcessControlBlock::new(heap, heap_size);
    tenuring_gc_test(process, false);
}

// This test is a further extension of `gc_minor_tenuring_test` that ensures that a full sweep
// which occurs after tenuring has occured, results in all tenured objects being moved to a fresh
// young generation heap, with the old generation heap having been freed
#[test]
#[ignore]
fn gc_fullsweep_after_tenuring_test() {
    let (heap, heap_size) = alloc::default_heap().unwrap();
    let process = ProcessControlBlock::new(heap, heap_size);
    tenuring_gc_test(process, true);
}

fn simple_gc_test(mut process: ProcessControlBlock) {
    // Allocate an `{:ok, "hello world"}` tuple
    // First, the `ok` atom, an immediate, is super easy
    let ok = unsafe { Atom::try_from_str("ok").unwrap().as_term() };
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_term = make_binary_from_str(&mut process, greeting).unwrap();
    // Finally, allocate room for the tuple itself, which is essentially an
    // array of `Term`, which in the case of immediates actually _is_ an array,
    // but as in our test here, when boxed terms are involved, doesn't fully
    // contain everything.
    let elements = [ok, greeting_term];
    let tuple_term = make_tuple_from_slice(&mut process, &elements).unwrap();
    assert!(tuple_term.is_boxed());
    let tuple_ptr = tuple_term.boxed_val();

    // Allocate the list `[101, "test"]`
    let num = make_integer(&mut process, 101usize);
    let string = "test";
    let string_term = make_binary_from_str(&mut process, string).unwrap();
    let list_term = ListBuilder::on_heap(&mut process)
        .push(num)
        .push(string_term)
        .finish()
        .unwrap();
    assert!(list_term.is_list());
    let list_ptr = list_term.list_val();

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
    let roots = [tuple_term, list_term];
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
    // Assert that roots have been updated, as the underlying object should have moved to a new heap
    // First the tuple
    let tuple_root = roots[0];
    assert!(tuple_root.is_boxed());
    let new_tuple_ptr = follow_moved(tuple_root).boxed_val();
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
    assert_eq!("goodbye!", greeting_str.as_str());

    let list_root = roots[1];
    assert!(list_root.is_list());
    let new_list_ptr = follow_moved(list_root).list_val();
    assert_ne!(new_list_ptr, list_ptr);
    // Assert that we can still access list elements
    let new_list = unsafe { &*new_list_ptr };
    // The first value should be the integer
    assert!(new_list.head.is_smallint());
    // The tail should be another cons cell
    assert!(new_list.tail.is_list());
    let new_list_tail = unsafe { &*new_list.tail.list_val() };
    // The last value should be a heapbin
    assert!(new_list_tail.head.is_boxed());
    let test_string_term_ptr = new_list_tail.head.boxed_val();
    let test_string_term = unsafe { *test_string_term_ptr };
    assert!(test_string_term.is_heapbin());
    let test_string = unsafe { &*(test_string_term_ptr as *mut HeapBin) };
    assert_eq!("test", test_string.as_str());
}

fn tenuring_gc_test(mut process: ProcessControlBlock, perform_fullsweep: bool) {
    // Allocate an `{:ok, "hello world"}` tuple
    // First, the `ok` atom, an immediate, is super easy
    let ok = unsafe { Atom::try_from_str("ok").unwrap().as_term() };
    // Second, the binary, which will be a HeapBin (under 64 bytes),
    // requires space to be allocated for the header as well as the contents,
    // then have both written to the heap
    let greeting = "hello world";
    let greeting_term = make_binary_from_str(&mut process, greeting).unwrap();
    // Construct tuple containing the atom and string
    let elements = [ok, greeting_term];
    let tuple_term = make_tuple_from_slice(&mut process, &elements).unwrap();
    // Verify that the resulting tuple is valid
    assert!(tuple_term.is_boxed());
    let tuple_ptr = tuple_term.boxed_val();
    let tup = unsafe { &mut *(tuple_ptr as *mut Tuple) };
    let mut tup_iter = tup.iter();
    let t1 = tup_iter.next().unwrap();
    dbg!(t1);
    assert!(t1.is_atom());
    let t2 = tup_iter.next().unwrap();
    dbg!(t2);
    assert!(t2.is_boxed());
    let t2ptr = t2.boxed_val();
    let t2val = unsafe { *t2ptr };
    dbg!(t2val);
    assert!(t2val.is_heapbin());
    let t2str = unsafe { &*(t2ptr as *mut HeapBin) };
    assert_eq!(t2str.as_str(), greeting);

    // Allocate a list `[101, "this is a list"]`
    let num = Term::make_smallint(101);
    let string = "this is a list";
    let string_term = make_binary_from_str(&mut process, string).unwrap();
    let list_term = ListBuilder::on_heap(&mut process)
        .push(num)
        .push(string_term)
        .finish()
        .unwrap();
    assert!(list_term.is_list());
    // Verify the resulting list is valid
    assert!(list_term.is_list());
    let cons_ptr = list_term.list_val();
    let cons = unsafe { &*cons_ptr };
    let mut cons_iter = cons.iter();
    let l1 = cons_iter.next().unwrap();
    dbg!(l1);
    assert!(l1.is_smallint());
    let l2 = cons_iter.next().unwrap();
    dbg!(l2);
    assert!(l2.is_boxed());
    let l2ptr = l2.boxed_val();
    let l2val = unsafe { *l2ptr };
    dbg!(l2val);
    assert!(l2val.is_heapbin());
    let l2str = unsafe { &*(l2ptr as *mut HeapBin) };
    assert_eq!(l2str.as_str(), string);

    // Put term references on the stack
    process.stack_push(tuple_term).unwrap();
    process.stack_push(list_term).unwrap();

    // Now, we will simulate updating the greeting of the above tuple with a new one,
    // leaving the original greeting dead, and a target for collection
    let new_greeting = "goodbye world!";
    let new_greeting_term = make_binary_from_str(&mut process, new_greeting).unwrap();

    // Update second element of the tuple above
    tup.set_element_internal(2, new_greeting_term).unwrap();
    let mut tup_iter = tup.iter();
    let t1 = tup_iter.next().unwrap();
    assert!(t1.is_smallint());
    let t2 = tup_iter.next().unwrap();
    assert!(t2.is_boxed());
    let t2ptr = t2.boxed_val();
    let t2val = unsafe { *t2ptr };
    assert!(t2val.is_heapbin());
    let t2str = unsafe { &*(t2ptr as *mut HeapBin) };
    assert_eq!(t2str.as_str(), new_greeting);

    // Grab current heap size
    let peak_size = process.young.heap_used();
    // Run first garbage collection
    let roots = [];
    process.garbage_collect(0, &roots).unwrap();

    // Verify size of garbage collected meets expectation
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

    // Verify roots, starting with the list since it is on top of the stack
    let list_term = process.stack_pop().unwrap();
    assert!(list_term.is_list());
    let list_term_ptr = list_term.list_val();
    let list = unsafe { &*list_term_ptr };
    assert!(!list.is_move_marker());
    let mut list_iter = list.iter();
    let l0 = list_iter.next().unwrap();
    assert_eq!(l0, Term::make_smallint(101));
    let l1 = list_iter.next().unwrap();
    assert!(l1.is_boxed());
    let l1ptr = l1.boxed_val();
    let l1bin = unsafe { *l1ptr };
    assert!(l1bin.is_heapbin());
    let l1str = unsafe { &*(l1ptr as *mut HeapBin) };
    assert_eq!(l1str.as_str(), string);
    assert_eq!(list_iter.next(), None);

    let tuple_term = process.stack_pop().unwrap();
    let tuple_ptr_postgc = tuple_term.boxed_val();
    assert_eq!(tuple_ptr_postgc, tuple_ptr);
    let tuple_boxed = unsafe { *tuple_ptr };
    assert!(is_move_marker(tuple_boxed));
    let moved_tuple_ptr = tuple_boxed.boxed_val();
    assert_ne!(moved_tuple_ptr, tuple_ptr);
    assert_ne!(moved_tuple_ptr, tuple_ptr_postgc);
    let moved_tuple = unsafe { *moved_tuple_ptr };
    assert!(moved_tuple.is_tuple_with_arity(2));
    let tup = unsafe { &*(moved_tuple_ptr as *mut Tuple) };
    let mut tup_iter = tup.iter();
    let t1 = tup_iter.next().unwrap();
    assert!(t1.is_smallint());
    let t2 = tup_iter.next().unwrap();
    assert!(t2.is_boxed());
    let t2ptr = t2.boxed_val();
    let t2val = unsafe { *t2ptr };
    assert!(t2val.is_heapbin());
    let t2str = unsafe { &*(t2ptr as *mut HeapBin) };
    assert_eq!(t2str.as_str(), new_greeting);

    // Push tuple back on stack, resolving the move marker
    process.stack_push(moved_tuple).unwrap();

    // Allocate a fresh list for the young generation which references the older list,
    // e.g. will be equivalent to `[202, 101, "this is a list"]
    let num2 = Term::make_smallint(202);
    let second_list_term = ListBuilder::on_heap(&mut process)
        .push(num2)
        .push(list_term)
        .finish()
        .unwrap();
    let second_list_ptr = second_list_term.list_val();
    assert!(list_term.is_list());
    let second_list = unsafe { &*second_list_ptr };
    let mut list_iter = second_list.iter();
    let l0 = list_iter.next().unwrap();
    assert_eq!(l0, Term::make_smallint(202));
    let l1 = list_iter.next().unwrap();
    assert_eq!(l1, Term::make_smallint(101));
    let l2 = list_iter.next().unwrap();
    assert!(l2.is_boxed());
    let l2ptr = l2.boxed_val();
    let l2bin = unsafe { *l2ptr };
    assert!(l2bin.is_heapbin());
    let l1str = unsafe { &*(l2ptr as *mut HeapBin) };
    assert_eq!(l2str.as_str(), string);
    assert_eq!(list_iter.next(), None);

    // Push reference to new list on stack
    process.stack_push(second_list_term).unwrap();

    // Run second garbage collection, which should tenure everything except the new term we just allocated
    let second_peak_size = process.young.heap_used();
    let roots = [];
    process.garbage_collect(0, &roots).unwrap();

    // Verify no garbage was collected, we should have just tenured some data,
    // the only data on the young heap should be a single cons cell
    dbg!(&process.young);
    dbg!(&process.old);
    let second_collected_size = process.young.heap_used();
    let newly_allocated_size = to_word_size(mem::size_of::<Cons>());
    assert_eq!(second_collected_size, newly_allocated_size);
    assert_eq!(second_peak_size + newly_allocated_size, second_collected_size);

    // TODO

    // Verify that we now have an old generation and that it is of the expected size
    assert!(process.old.active());
    assert_eq!(process.old.heap_used(), collected_size, "expected tenuring of older young generation heap");

    // Verify roots
    let list_term = process.stack_pop().unwrap();
    assert!(list_term.is_list());
    let list_term_ptr = list_term.list_val();
    let list = unsafe { &*list_term_ptr };
    assert!(!list.is_move_marker());
    let mut list_iter = list.iter();
    let l0 = list_iter.next().unwrap();
    assert!(l0.is_smallint());
    let l1 = list_iter.next().unwrap();
    assert!(l1.is_smallint());
    let l2 = list_iter.next().unwrap();
    assert!(l2.is_boxed());
    let l2_bin = unsafe { *l2.boxed_val() };
    assert!(l2_bin.is_heapbin());
    assert_eq!(list_iter.next(), None);

    let tuple_root = process.stack_pop().unwrap();
    verify_tuple_root(tuple_root, tuple_ptr)

    // Assert that roots have been updated properly
    /*
    let list_root = process.stack_pop().unwrap();
    assert!(list_root.is_list());
    let new_list_ptr = list_root.list_val();
    assert_ne!(new_list_ptr, second_list_ptr);
    // Assert that we can still access list elements
    let new_list = unsafe { &*new_list_ptr };
    assert!(!new_list.is_move_marker());
    // The first value should be the integer
    assert!(new_list.head.is_smallint());
    // The tail should be another cons cell
    assert!(new_list.tail.is_list());
    let new_list_tail = unsafe { &*new_list.tail.list_val() };
    // The last value should be a heapbin
    assert!(new_list_tail.head.is_boxed());
    let test_string_term_ptr = new_list_tail.head.boxed_val();
    let test_string_term = unsafe { *test_string_term_ptr };
    assert!(test_string_term.is_heapbin());
    let test_string = unsafe { &*(test_string_term_ptr as *mut HeapBin) };
    assert_eq!("test", test_string.as_str());
    */
}

fn verify_tuple_root(tuple_root: Term, tuple_ptr: *mut Term) {
    assert!(tuple_root.is_boxed());
    let new_tuple_ptr = tuple_root.boxed_val();
    assert_ne!(new_tuple_ptr, tuple_ptr as *mut Term);
    let new_tuple_term = unsafe { *new_tuple_ptr };
    assert!(!is_move_marker(new_tuple_term));
    // Assert that we can still access data that should be live
    let new_tuple = unsafe { &*(new_tuple_ptr as *mut Tuple) };
    assert_eq!(new_tuple.size(), 2);
    // First, the atom
    let ok = unsafe { Atom::try_from_str("ok").unwrap().as_term() };
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
    assert_eq!("goodbye!", greeting_str.as_str());
}