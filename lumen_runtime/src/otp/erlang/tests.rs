use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};
use crate::exception::Result;
use crate::otp::erlang;

mod abs_1;
mod add_2;
mod append_element_2;
mod atom_to_binary_2;
mod atom_to_list_1;
mod binary_part_2;
mod binary_part_3;
mod binary_to_atom_2;
mod binary_to_existing_atom_2;
mod binary_to_float_1;
mod binary_to_integer_1;
mod binary_to_integer_2;
mod binary_to_list_1;
mod binary_to_list_3;
mod binary_to_term_1;
mod binary_to_term_2;
mod bit_size_1;
mod bitstring_to_list_1;
mod byte_size_1;
mod ceil_1;
mod concatenate_2;
mod convert_time_unit_3;
mod delete_element_2;
mod div_2;
mod divide_2;
mod element_2;
mod error_1;
mod error_2;
mod exit_1;
mod hd_1;
mod insert_element_3;
mod is_alive_0;
mod is_atom_1;
mod is_binary_1;
mod is_bitstring_1;
mod is_boolean_1;
mod is_float_1;
mod is_integer_1;
mod is_list_1;
mod is_map_1;
mod is_map_key_2;
mod is_number_1;
mod is_pid_1;
mod is_record_2;
mod is_record_3;
mod is_reference_1;
mod is_tuple_1;
mod length_1;
mod list_to_atom_1;
mod list_to_binary_1;
mod list_to_bitstring_1;
mod list_to_existing_atom_1;
mod list_to_pid_1;
mod list_to_tuple_1;
mod make_ref_0;
mod map_get_2;
mod map_size_1;
mod multiply_2;
mod negate_1;
mod node_0;
mod number_or_badarith_1;
mod raise_3;
mod rem_2;
mod self_0;
mod setelement_3;
mod size_1;
mod split_binary_2;
mod subtract_2;
mod subtract_list_2;
mod throw_1;
mod tl_1;
mod tuple_size_1;
mod tuple_to_list_1;

fn errors_badarg<F>(actual: F)
where
    F: FnOnce(&mut Process) -> Result,
{
    with_process(|mut process| assert_badarg!(actual(&mut process)))
}

fn errors_badarith<F>(actual: F)
where
    F: FnOnce(&mut Process) -> Result,
{
    with_process(|mut process| assert_badarith!(actual(&mut process)))
}

fn list_term(process: &mut Process) -> Term {
    let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
    Term::cons(head_term, Term::EMPTY_LIST, process)
}

fn with_process<F>(f: F)
where
    F: FnOnce(&mut Process) -> (),
{
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    f(&mut process)
}
