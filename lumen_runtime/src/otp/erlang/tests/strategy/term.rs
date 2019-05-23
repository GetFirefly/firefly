use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use crate::atom::Existence::DoNotCare;
use crate::process::{IntoProcess, ModuleFunctionArity, Process};
use crate::term::Term;

pub mod integer;

pub fn atom() -> BoxedStrategy<Term> {
    any::<String>()
        .prop_map(|s| Term::str_to_atom(&s, DoNotCare).unwrap())
        .boxed()
}

pub fn container(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    prop_oneof![
        tuple(element.clone(), size_range.clone(), arc_process.clone()),
        map(element.clone(), size_range.clone(), arc_process.clone()),
        list(element, size_range.clone(), arc_process.clone())
    ]
}

pub fn float(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<f64>()
        .prop_map(move |f| f.into_process(&arc_process))
        .boxed()
}

pub fn function(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
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

pub fn heap_binary(size_range: SizeRange, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::byte_vec(size_range)
        .prop_map(move |byte_vec| Term::slice_to_binary(&byte_vec, &arc_process))
        .boxed()
}

pub fn is_not_number(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a number", |v| !v.is_number())
        .boxed()
}

pub fn is_number(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must be a number", |v| v.is_number())
        .boxed()
}

pub fn leaf(
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
        integer::big(big_integer_arc_process),
        local_reference(local_reference_arc_process),
        function(function_arc_process),
        float(float_arc_process),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        heap_binary(heap_binary_size_range, heap_binary_arc_process),
        subbinary(subbinary_range_inclusive, subbinary_arc_process),
        // TODO `ExternalPid`
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::EMPTY_LIST),
        local_pid(),
        // TODO `LocalPort`,
        atom(),
        integer::small(small_integer_arc_process)
    ]
    .boxed()
}

pub fn list(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_list(&vec, &arc_process))
        .boxed()
}

pub fn local_pid() -> BoxedStrategy<Term> {
    (
        0..crate::process::identifier::NUMBER_MAX,
        0..crate::process::identifier::SERIAL_MAX,
    )
        .prop_map(|(number, serial)| Term::local_pid(number, serial).unwrap())
        .boxed()
}

fn local_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    proptest::prelude::any::<u64>()
        .prop_map(move |number| Term::local_reference(number, &arc_process))
        .boxed()
}

fn map(
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

fn subbinary(bit_range: RangeInclusive<usize>, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let arc_process_clone1 = arc_process.clone();
    let arc_process_clone2 = arc_process.clone();

    bit_range
        .prop_flat_map(move |bit_count| {
            let byte_size = super::bits_to_bytes(bit_count);

            (
                Just(bit_count),
                heap_binary((byte_size..=byte_size).into(), arc_process_clone1.clone()),
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

fn tuple(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    proptest::collection::vec(element, size_range)
        .prop_map(move |vec| Term::slice_to_tuple(&vec, &arc_process))
        .boxed()
}
