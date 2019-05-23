use super::*;

use std::ops::RangeInclusive;

use proptest::arbitrary::any;
use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq, prop_oneof};

use crate::process::ModuleFunctionArity;

mod with_big_integer_augend;
mod with_float_augend;
mod with_small_integer_augend;

#[test]
fn without_number_augend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    term_is_not_number_strategy(arc_process.clone()),
                    term_is_number_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn atom_term_strategy() -> BoxedStrategy<Term> {
    any::<String>()
        .prop_map(|s| Term::str_to_atom(&s, DoNotCare).unwrap())
        .boxed()
}

fn big_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        negative_big_integer_term_strategy(arc_process.clone()),
        positive_big_integer_term_strategy(arc_process)
    ]
    .boxed()
}

fn negative_big_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (std::isize::MIN..(integer::small::MIN - 1))
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

fn positive_big_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    ((integer::small::MAX + 1)..std::isize::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

fn float_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<f64>()
        .prop_map(move |f| f.into_process(&arc_process))
        .boxed()
}

fn function_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let module = Term::str_to_atom("module", DoNotCare).unwrap();
    let function = Term::str_to_atom("function", DoNotCare).unwrap();
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module,
        function,
        arity: 0,
    });
    let code = |arc_process: &Arc<Process>| arc_process.wait();

    Just(Term::function(module_function_arity, code, &arc_process)).boxed()
}

fn byte_vec_strategy(size_range: SizeRange) -> BoxedStrategy<Vec<u8>> {
    proptest::collection::vec(proptest::prelude::any::<u8>(), size_range).boxed()
}

fn heap_binary_term_strategy(
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    byte_vec_strategy(size_range)
        .prop_map(move |byte_vec| Term::slice_to_binary(&byte_vec, &arc_process))
        .boxed()
}

fn list_term_strategy(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_list(&vec, &arc_process))
        .boxed()
}

fn local_pid_term_strategy() -> BoxedStrategy<Term> {
    (
        0..process::identifier::NUMBER_MAX,
        0..process::identifier::SERIAL_MAX,
    )
        .prop_map(|(number, serial)| Term::local_pid(number, serial).unwrap())
        .boxed()
}

fn local_reference_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    proptest::prelude::any::<u64>()
        .prop_map(move |number| Term::local_reference(number, &arc_process))
        .boxed()
}

fn map_term_strategy(
    key_or_value: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::hash_map(key_or_value.clone(), key_or_value, size_range)
        .prop_map(move |mut hash_map| {
            let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

            Term::slice_to_map(&entry_vec, &arc_process)
        })
        .boxed()
}

fn small_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (integer::small::MIN..integer::small::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

#[allow(dead_code)]
fn negative_small_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (integer::small::MIN..-1)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

fn positive_small_integer_term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (1..integer::small::MAX)
        .prop_map(move |i| i.into_process(&arc_process))
        .boxed()
}

fn bits_to_bytes(bits: usize) -> usize {
    (bits + 7) / 8
}

fn subbinary_term_strategy(
    bit_range: RangeInclusive<usize>,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    let arc_process_clone1 = arc_process.clone();
    let arc_process_clone2 = arc_process.clone();

    bit_range
        .prop_flat_map(move |bit_count| {
            let byte_size = bits_to_bytes(bit_count);

            (
                Just(bit_count),
                heap_binary_term_strategy(
                    (byte_size..=byte_size).into(),
                    arc_process_clone1.clone(),
                ),
            )
        })
        .prop_map(move |(bit_count, original)| {
            // TODO vary byte_offset, bit_offset, byte_count, and bit_count
            Term::subbinary(
                original,
                0,
                0,
                bit_count / 8,
                (bit_count % 8) as u8,
                &arc_process_clone2.clone(),
            )
        })
        .boxed()
}

fn term_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let max_len = 16;
    let size_range_inclusive = 0..=max_len;
    let size_range: SizeRange = size_range_inclusive.clone().into();

    let container_arc_process = arc_process.clone();

    term_leaf_strategy(size_range_inclusive, arc_process)
        .prop_recursive(4, 64, 16, move |element| {
            term_container_strategy(element, size_range.clone(), container_arc_process.clone())
        })
        .boxed()
}

fn term_is_number_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    term_strategy(arc_process)
        .prop_filter("Value must be a number", |v| v.is_number())
        .boxed()
}

fn term_is_not_number_strategy(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    term_strategy(arc_process)
        .prop_filter("Value must not be a number", |v| !v.is_number())
        .boxed()
}

fn term_container_strategy(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    prop_oneof![
        tuple_term_strategy(element.clone(), size_range.clone(), arc_process.clone()),
        map_term_strategy(element.clone(), size_range.clone(), arc_process.clone()),
        list_term_strategy(element, size_range.clone(), arc_process.clone())
    ]
}

fn term_leaf_strategy(
    range_inclusive: RangeInclusive<usize>,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    let big_integer_arc_process = arc_process.clone();
    let local_reference_arc_process = arc_process.clone();
    let function_arc_process = arc_process.clone();
    let float_arc_process = arc_process.clone();

    let heap_binary_arc_process = arc_process.clone();
    let heap_binary_size_range = range_inclusive.clone().into();

    let subbinary_arc_process = arc_process.clone();
    let subbinary_range_inclusive = range_inclusive.clone();

    let small_integer_arc_process = arc_process.clone();

    prop_oneof![
        // TODO `BinaryAggregate`
        big_integer_term_strategy(big_integer_arc_process),
        local_reference_term_strategy(local_reference_arc_process),
        function_term_strategy(function_arc_process),
        float_term_strategy(float_arc_process),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        heap_binary_term_strategy(heap_binary_size_range, heap_binary_arc_process),
        subbinary_term_strategy(subbinary_range_inclusive, subbinary_arc_process),
        // TODO `ExternalPid`
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::EMPTY_LIST),
        local_pid_term_strategy(),
        // TODO `LocalPort`,
        atom_term_strategy(),
        small_integer_term_strategy(small_integer_arc_process)
    ]
    .boxed()
}

fn tuple_term_strategy(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_tuple(&vec, &arc_process))
        .boxed()
}
