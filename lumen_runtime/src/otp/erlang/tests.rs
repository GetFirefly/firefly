use super::*;

use crate::otp::erlang;

mod abs;
mod append_element;
mod atom_to_binary;
mod atom_to_list;
mod binary_byte_range_to_list;
mod binary_in_base_to_integer;
mod binary_options_to_term;
mod binary_part;
mod binary_to_atom;
mod binary_to_existing_atom;
mod binary_to_float;
mod binary_to_integer;
mod binary_to_list;
mod binary_to_term;
mod bit_size;
mod bitstring_to_list;
mod byte_size;
mod ceil;
mod convert_time_unit;
mod delete_element;
mod element;
mod error;
mod error_with_arguments;
mod head;
mod insert_element;
mod is_atom;
mod is_binary;
mod is_boolean;
mod is_float;
mod is_integer;
mod is_list;
mod is_map;
mod is_map_key;
mod is_number;
mod is_pid;
mod is_record;
mod is_tuple;
mod length;
mod list_to_pid;
mod self_pid;
mod size;
mod tail;

fn list_term(mut process: &mut Process) -> Term {
    let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
    Term::cons(head_term, Term::EMPTY_LIST, process)
}
