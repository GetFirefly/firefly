use std::cmp::{max, min};
use std::num::FpCategory;
use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use crate::atom::Existence::DoNotCare;
use crate::process::{IntoProcess, ModuleFunctionArity, Process};
use crate::term::Tag::Atom;
use crate::term::Term;

use super::{size_range, DEPTH, MAX_LEN, RANGE_INCLUSIVE};

pub mod binary;
pub mod container;
pub mod function;
pub mod integer;
pub mod is_binary;
pub mod is_bitstring;
pub mod list;
pub mod map;
pub mod pid;
pub mod tuple;

pub const NON_EXISTENT_ATOM_PREFIX: &str = "non_existent";

pub fn atom() -> BoxedStrategy<Term> {
    any::<String>()
        .prop_filter("Reserved for existing/safe atom tests", |s| {
            !s.starts_with(NON_EXISTENT_ATOM_PREFIX)
        })
        .prop_map(|s| Term::str_to_atom(&s, DoNotCare).unwrap())
        .boxed()
}

/// Produces `i64` that fall in the range that produce both integral floats and big integers
pub fn big_integer_float_integral_i64() -> Option<BoxedStrategy<i64>> {
    negative_big_integer_float_integral_i64().and_then(|negative| {
        match positive_big_integer_float_integral_i64() {
            Some(positive) => Some(negative.prop_union(positive).boxed()),
            None => None,
        }
    })
}

// XXX work-around for bug related to sending maps across processes in `send_2` proptests
pub fn can_be_passed_to_different_process(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let container_arc_process = arc_process.clone();

    leaf(RANGE_INCLUSIVE, arc_process)
        .prop_recursive(
            DEPTH,
            (MAX_LEN * (DEPTH as usize + 1)) as u32,
            MAX_LEN as u32,
            move |element| {
                container::can_be_passed_to_different_process(
                    element,
                    RANGE_INCLUSIVE.clone().into(),
                    container_arc_process.clone(),
                )
            },
        )
        .boxed()
}

pub fn charlist(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<String>()
        .prop_map(move |string| {
            let codepoint_terms: Vec<Term> = string
                .chars()
                .map(|c| c.into_process(&arc_process))
                .collect();

            Term::slice_to_list(&codepoint_terms, &arc_process)
        })
        .boxed()
}

pub fn container(
    element: BoxedStrategy<Term>,
    size_range: SizeRange,
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        tuple::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
        map::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
        list::intermediate(element, size_range.clone(), arc_process.clone())
    ]
    .boxed()
}

pub fn float(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<f64>()
        .prop_filter("Negative and positive 0.0 are the same for Erlang", |f| {
            !(f.classify() == FpCategory::Zero && f.is_sign_negative())
        })
        .prop_map(move |f| f.into_process(&arc_process))
        .boxed()
}

pub fn function(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        function::module(),
        function::function(),
        function::arity_usize(),
    )
        .prop_map(move |(module, function, arity)| {
            let module_function_arity = Arc::new(ModuleFunctionArity {
                module,
                function,
                arity,
            });
            let code = |arc_process: &Arc<Process>| arc_process.wait();

            Term::function(module_function_arity, code, &arc_process)
        })
        .boxed()
}

pub fn is_binary(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        binary::heap(arc_process.clone()),
        binary::sub::is_binary(arc_process)
    ]
    .boxed()
}

pub fn is_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![binary::heap(arc_process.clone()), binary::sub(arc_process)].boxed()
}

pub fn is_boolean() -> BoxedStrategy<Term> {
    prop_oneof![Just(true.into()), Just(false.into())].boxed()
}

pub fn is_encoding() -> BoxedStrategy<Term> {
    prop_oneof![
        Just(Term::str_to_atom("latin1", DoNotCare).unwrap()),
        Just(Term::str_to_atom("unicode", DoNotCare).unwrap()),
        Just(Term::str_to_atom("utf8", DoNotCare).unwrap())
    ]
    .boxed()
}

pub fn is_integer(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        integer::small(arc_process.clone()),
        integer::big(arc_process)
    ]
    .boxed()
}

pub fn is_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    list::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn is_map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    map::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn is_not_atom(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be an atom", |v| !v.is_atom())
        .boxed()
}

pub fn is_not_binary(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = super::term(arc_process.clone());
    let size_range = super::size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::EMPTY_LIST),
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(arc_process.clone()),
        container(element.clone(), size_range, arc_process.clone())
    ]
    .boxed()
}

pub fn is_not_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = super::term(arc_process.clone());
    let size_range = super::size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::EMPTY_LIST),
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(arc_process.clone()),
        container(element.clone(), size_range, arc_process.clone())
    ]
    .boxed()
}

pub fn is_not_boolean(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Atom cannot be a boolean", |v| !v.is_boolean())
        .boxed()
}

pub fn is_not_encoding(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter(
            "Must either not be an atom or not be an atom encoding atom",
            |term| {
                term.tag() != Atom || {
                    match unsafe { term.atom_to_string() }.as_ref().as_ref() {
                        "latin1" | "unicode" | "utf8" => false,
                        _ => true,
                    }
                }
            },
        )
        .boxed()
}

pub fn is_not_float(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a float", |v| !v.is_float())
        .boxed()
}

pub fn is_not_integer(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be an integer", |v| !v.is_integer())
        .boxed()
}

pub fn is_not_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = super::term(arc_process.clone());
    let size_range = super::size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        binary::heap::with_size_range(size_range.clone(), arc_process.clone()),
        binary::sub(arc_process.clone()),
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(arc_process.clone()),
        prop_oneof![
            tuple::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
            map::intermediate(element.clone(), size_range, arc_process.clone()),
        ]
    ]
    .boxed()
}

pub fn is_not_map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a map", |v| !v.is_map())
        .boxed()
}

pub fn is_not_number(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a number", |v| !v.is_number())
        .boxed()
}

pub fn is_not_pid(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a pid", |v| !v.is_pid())
        .boxed()
}

pub fn is_not_proper_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = super::term(arc_process.clone());
    let size_range = super::size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        binary::heap::with_size_range(size_range.clone(), arc_process.clone()),
        binary::sub(arc_process.clone()),
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(arc_process.clone()),
        prop_oneof![
            tuple::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
            map::intermediate(element.clone(), size_range, arc_process.clone()),
            list::improper(arc_process.clone())
        ]
    ]
    .boxed()
}

pub fn is_not_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a tuple", |v| !v.is_reference())
        .boxed()
}

pub fn is_not_tuple(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a tuple", |v| !v.is_tuple())
        .boxed()
}

// `super::term(arc_process).prop_filter(..., |v| v.is_number())` is too slow, on the order of
// minutes instead of seconds because most terms aren't numbers, so this directly uses the
// number strategies instead.
pub fn is_number(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let big_integer_arc_process = arc_process.clone();
    let float_arc_process = arc_process.clone();
    let small_integer_arc_process = arc_process.clone();

    prop_oneof![
        integer::big(big_integer_arc_process),
        float(float_arc_process),
        integer::small(small_integer_arc_process)
    ]
    .boxed()
}

pub fn is_pid(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![pid::external(arc_process.clone()), pid::local()].boxed()
}

pub fn is_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        local_reference(arc_process),
        // TODO `ExternalReference`
    ]
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

    let external_pid_arc_process = arc_process.clone();

    let small_integer_arc_process = arc_process.clone();

    prop_oneof![
        // TODO `BinaryAggregate`
        integer::big(big_integer_arc_process),
        local_reference(local_reference_arc_process),
        function(function_arc_process),
        float(float_arc_process),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        binary::heap::with_size_range(heap_binary_size_range, heap_binary_arc_process),
        binary::sub(subbinary_arc_process),
        pid::external(external_pid_arc_process),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::EMPTY_LIST),
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(small_integer_arc_process)
    ]
    .boxed()
}

pub fn local_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    proptest::prelude::any::<u64>()
        .prop_map(move |number| Term::local_reference(number, &arc_process))
        .boxed()
}

pub fn map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    map::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

fn negative_big_integer_float_integral_i64() -> Option<BoxedStrategy<i64>> {
    let float_integral_min = crate::float::INTEGRAL_MIN as i64;
    let big_integer_max_negative = crate::integer::small::MIN as i64 - 1;

    if float_integral_min < big_integer_max_negative {
        let boxed_strategy: BoxedStrategy<i64> =
            (float_integral_min..=big_integer_max_negative).boxed();

        Some(boxed_strategy)
    } else {
        None
    }
}

pub fn non_existent_atom(suffix: &str) -> String {
    format!("{}_{}", NON_EXISTENT_ATOM_PREFIX, suffix)
}

pub fn pid_or_port(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        pid::local(),
        // TODO `LocalPort`,
    ]
    .boxed()
}

fn positive_big_integer_float_integral_i64() -> Option<BoxedStrategy<i64>> {
    let float_integral_max = crate::float::INTEGRAL_MAX as i64;
    let big_integer_min_positive = crate::integer::small::MAX as i64 + 1;

    if big_integer_min_positive < float_integral_max {
        let boxed_strategy: BoxedStrategy<i64> =
            (big_integer_min_positive..=float_integral_max).boxed();

        Some(boxed_strategy)
    } else {
        None
    }
}

/// Produces `i64` that fall in the range that produce both integral floats and small integers
pub fn small_integer_float_integral_i64() -> BoxedStrategy<i64> {
    let integral_min = max(
        crate::float::INTEGRAL_MIN as i64,
        crate::integer::small::MIN as i64,
    );
    let integral_max = min(
        crate::float::INTEGRAL_MAX as i64,
        crate::integer::small::MAX as i64,
    );

    (integral_min..=integral_max).boxed()
}

pub fn tuple(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    tuple::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}
