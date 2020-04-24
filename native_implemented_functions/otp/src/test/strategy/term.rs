use std::cmp::{max, min};
use std::convert::TryInto;
use std::ffi::c_void;
use std::num::FpCategory;
use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::collection::SizeRange;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;
use liblumen_alloc::{atom, fixnum_from};

use crate::runtime::process::current_process;

use super::size_range;

pub mod atom;
pub mod binary;
pub mod function;
pub mod index;
pub mod integer;
pub mod is_binary;
pub mod is_bitstring;
pub mod list;
pub mod map;
pub mod pid;
pub mod tuple;

pub const NON_EXISTENT_ATOM_PREFIX: &str = "non_existent";

pub fn atom() -> BoxedStrategy<Term> {
    super::atom()
        .prop_map(|atom| atom.encode().unwrap())
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

pub fn charlist(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<String>()
        .prop_map(move |string| {
            let codepoint_terms: Vec<Term> = string
                .chars()
                .map(|c| fixnum_from!(c as u32))
                .map(|f| f.into())
                .collect();

            arc_process.list_from_slice(&codepoint_terms).unwrap()
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
        /*        map::intermediate(element.clone(), size_range.clone(), arc_process.clone()),
         *        list::intermediate(element, size_range.clone(), arc_process.clone()) */
    ]
    .boxed()
}

pub fn float(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<f64>()
        .prop_filter("Negative and positive 0.0 are the same for Erlang", |f| {
            !(f.classify() == FpCategory::Zero && f.is_sign_negative())
        })
        .prop_map(move |f| arc_process.float(f).unwrap())
        .boxed()
}

pub fn function_port_pid_tuple_map_list_or_bitstring(
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        is_function(arc_process.clone()),
        // TODO `Port` and `ExternalPort`
        is_pid(arc_process.clone()),
        tuple(arc_process.clone()),
        map(arc_process.clone()),
        is_bitstring(arc_process.clone()),
    ]
    .boxed()
}

pub fn is_base(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::base::base()
        .prop_map(move |base| arc_process.integer(base).unwrap())
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

pub fn is_byte(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (Just(arc_process), any::<u8>())
        .prop_map(|(arc_process, byte_u8)| arc_process.integer(byte_u8).unwrap())
        .boxed()
}

pub fn is_encoding() -> BoxedStrategy<Term> {
    prop_oneof![
        Just(atom!("latin1")),
        Just(atom!("unicode")),
        Just(atom!("utf8"))
    ]
    .boxed()
}

pub fn is_function(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        function::export(arc_process.clone()),
        function::anonymous(arc_process)
    ]
    .boxed()
}

pub fn is_function_with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    prop_oneof![
        function::export::with_arity(arc_process.clone(), arity),
        function::anonymous::with_arity(arc_process, arity)
    ]
    .boxed()
}

pub fn export_closure(process: &Process, module: Atom, function: Atom, arity: u8) -> Term {
    extern "C" fn zero() -> Term {
        let arc_process = current_process();
        arc_process.wait();

        Term::NONE
    }

    extern "C" fn one(_first: Term) -> Term {
        let arc_process = current_process();
        arc_process.wait();

        Term::NONE
    }

    extern "C" fn two(_first: Term, _second: Term) -> Term {
        let arc_process = current_process();
        arc_process.wait();

        Term::NONE
    }

    let native: *const c_void = match arity {
        0 => zero as _,
        1 => one as _,
        2 => two as _,
        _ => unimplemented!("Export closure with arity {}", arity),
    };

    process
        .export_closure(module, function, arity, Some(native as _))
        .unwrap()
}

pub fn export_closure_arity_range_inclusive() -> RangeInclusive<u8> {
    0..=2
}

pub fn export_closure_non_zero_arity_range_inclusive() -> RangeInclusive<u8> {
    1..=2
}

pub fn is_integer(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        integer::small(arc_process.clone()),
        integer::big(arc_process)
    ]
    .boxed()
}

pub fn is_iolist(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    list::io::root(arc_process)
}

pub fn is_iolist_or_binary(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![is_binary(arc_process.clone()), is_iolist(arc_process)].boxed()
}

pub fn is_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    list::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn is_map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    map::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn is_not_arity(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be 0-255", |term| {
            let result_u8: Result<u8, _> = (*term).try_into();

            result_u8.is_err()
        })
        .boxed()
}

pub fn is_not_atom(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be an atom", |v| !v.is_atom())
        .boxed()
}

pub(crate) fn is_not_base(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Cannot be a base (2-36)", |term| {
            match term.decode().unwrap() {
                TypedTerm::SmallInteger(small_integer) => {
                    let integer: isize = small_integer.into();

                    (2 <= integer) && (integer <= 36)
                }
                _ => true,
            }
        })
        .boxed()
}

pub fn is_not_binary(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let element = super::term(arc_process.clone());
    let size_range = super::size_range();

    prop_oneof![
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        is_function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::NIL),
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
        is_function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::NIL),
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

pub fn is_not_send_after_destination(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process.clone())
        .prop_filter(
            "send_after/start_timer destination must not be an atom or pid",
            |destination| !(destination.is_atom() || destination.is_pid()),
        )
        .boxed()
}

pub fn is_not_destination(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process.clone())
        .prop_filter(
            "Destination must not be an atom, pid, or tuple",
            |destination| {
                !(destination.is_atom() || destination.is_pid() || destination.is_boxed_tuple())
            },
        )
        .boxed()
}

pub fn is_not_float(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a float", |v| !v.is_boxed_float())
        .boxed()
}

pub fn is_not_function(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a function", |v| !v.is_boxed_function())
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
        is_function(arc_process.clone()),
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

pub fn is_not_local_pid(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a local pid", |term| !term.is_local_pid())
        .boxed()
}

pub fn is_not_local_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a local reference", |term| {
            !term.is_boxed_local_reference()
        })
        .boxed()
}

pub fn is_not_map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Term cannot be a map", |v| !v.is_boxed_map())
        .boxed()
}

pub fn is_not_non_negative_integer(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    let zero = arc_process.integer(0).unwrap();

    super::term(arc_process)
        .prop_filter("Term must no be a non-negative integer", move |term| {
            !(term.is_integer() && &zero <= term)
        })
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
        is_function(arc_process.clone()),
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
        .prop_filter("Value must not be a reference", |v| !v.is_reference())
        .boxed()
}

pub fn is_not_tuple(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    super::term(arc_process)
        .prop_filter("Value must not be a tuple", |v| !v.is_boxed_tuple())
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
    prop_oneof![
        // TODO `BinaryAggregate`
        integer::big(arc_process.clone()),
        local_reference(arc_process.clone()),
        is_function(arc_process.clone()),
        float(arc_process.clone()),
        // TODO `Export`
        // TODO `ReferenceCountedBinary`
        binary::heap::with_size_range(range_inclusive.into(), arc_process.clone()),
        binary::sub(arc_process.clone()),
        pid::external(arc_process.clone()),
        // TODO `ExternalPort`
        // TODO `ExternalReference`
        Just(Term::NIL),
        pid::local(),
        // TODO `LocalPort`,
        atom(),
        integer::small(arc_process.clone())
    ]
    .boxed()
}

pub fn list_or_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![is_list(arc_process.clone()), is_bitstring(arc_process)].boxed()
}

pub fn local_reference(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    proptest::prelude::any::<u64>()
        .prop_map(move |number| arc_process.reference(number).unwrap())
        .boxed()
}

pub fn map(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    map::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn map_list_or_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        map(arc_process.clone()),
        is_list(arc_process.clone()),
        is_bitstring(arc_process.clone())
    ]
    .boxed()
}

fn negative_big_integer_float_integral_i64() -> Option<BoxedStrategy<i64>> {
    let float_integral_min = Float::INTEGRAL_MIN as i64;
    let big_integer_max_negative = SmallInteger::MIN_VALUE as i64 - 1;

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

pub fn number_or_atom(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![is_number(arc_process), atom()].boxed()
}

pub fn number_atom_reference_function_or_port(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        is_number(arc_process.clone()),
        atom(),
        local_reference(arc_process.clone()),
        // TODO `ExternalReference`
        is_function(arc_process),
        // TODO Port
    ]
    .boxed()
}

pub fn number_atom_reference_function_port_or_local_pid(
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        is_number(arc_process.clone()),
        atom(),
        is_reference(arc_process.clone()),
        is_function(arc_process),
        // TODO ports
        pid::local()
    ]
    .boxed()
}

pub fn number_atom_reference_function_port_or_pid(
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        is_number(arc_process.clone()),
        atom(),
        is_reference(arc_process.clone()),
        is_function(arc_process.clone()),
        // TODO ports
        is_pid(arc_process)
    ]
    .boxed()
}

pub fn number_atom_reference_function_port_pid_or_tuple(
    arc_process: Arc<Process>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        is_number(arc_process.clone()),
        atom(),
        is_reference(arc_process.clone()),
        // TODO ports
        is_pid(arc_process.clone()),
        tuple(arc_process)
    ]
    .boxed()
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
    let float_integral_max = Float::INTEGRAL_MAX as i64;
    let big_integer_min_positive = SmallInteger::MAX_VALUE as i64 + 1;

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
    let integral_min = max(Float::INTEGRAL_MIN as i64, SmallInteger::MIN_VALUE as i64);
    let integral_max = min(Float::INTEGRAL_MAX as i64, SmallInteger::MAX_VALUE as i64);

    (integral_min..=integral_max).boxed()
}

pub fn tuple(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    tuple::intermediate(super::term(arc_process.clone()), size_range(), arc_process)
}

pub fn tuple_map_list_or_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        tuple(arc_process.clone()),
        is_map(arc_process.clone()),
        is_list(arc_process.clone()),
        is_bitstring(arc_process.clone())
    ]
    .boxed()
}
