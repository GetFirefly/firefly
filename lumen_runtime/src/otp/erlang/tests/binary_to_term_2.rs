use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

mod with_safe {
    use super::*;

    #[test]
    fn with_binary_encoding_atom_that_does_not_exist_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary(:atom)
        let binary_term = Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process);
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare, &mut process).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_bad_argument!(
            erlang::binary_to_term_2(binary_term, options, &mut process),
            &mut process
        );

        assert_eq_in_process!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::str_to_atom("atom", DoNotCare, &mut process).unwrap()),
            process
        );
    }

    #[test]
    fn with_binary_encoding_list_containing_atom_that_does_not_exist_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary([:atom])
        let binary_term = Term::slice_to_binary(
            &[131, 108, 0, 0, 0, 1, 100, 0, 4, 97, 116, 111, 109, 106],
            &mut process,
        );
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare, &mut process).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_bad_argument!(
            erlang::binary_to_term_2(binary_term, options, &mut process),
            &mut process
        );

        assert_eq_in_process!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::cons(
                Term::str_to_atom("atom", DoNotCare, &mut process).unwrap(),
                Term::EMPTY_LIST,
                &mut process
            )),
            process
        );
    }

    #[test]
    fn with_binary_encoding_small_tuple_containing_atom_that_does_not_exist_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary({:atom})
        let binary_term =
            Term::slice_to_binary(&[131, 104, 1, 100, 0, 4, 97, 116, 111, 109], &mut process);
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare, &mut process).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_bad_argument!(
            erlang::binary_to_term_2(binary_term, options, &mut process),
            &mut process
        );

        assert_eq_in_process!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::slice_to_tuple(
                &[Term::str_to_atom("atom", DoNotCare, &mut process).unwrap()],
                &mut process
            )),
            process
        );
    }

    #[test]
    fn with_binary_encoding_small_atom_utf8_that_does_not_exist_returns_bad_argument() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary(:"😈")
        let binary_term = Term::slice_to_binary(&[131, 119, 4, 240, 159, 152, 136], &mut process);
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare, &mut process).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_bad_argument!(
            erlang::binary_to_term_2(binary_term, options, &mut process),
            &mut process
        );

        assert_eq_in_process!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::str_to_atom("😈", DoNotCare, &mut process).unwrap()),
            process
        );
    }
}

#[test]
fn with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<131,100,0,5,"hello","world">>
    let binary_term = Term::slice_to_binary(
        &[
            131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100,
        ],
        &mut process,
    );
    let options = Term::cons(
        Term::str_to_atom("used", DoNotCare, &mut process).unwrap(),
        Term::EMPTY_LIST,
        &mut process,
    );

    let term = Term::str_to_atom("hello", DoNotCare, &mut process).unwrap();
    let result = erlang::binary_to_term_2(binary_term, options, &mut process);

    assert_eq_in_process!(
        result,
        Ok(Term::slice_to_tuple(
            &[term, 9.into_process(&mut process)],
            &mut process
        )),
        process
    );

    // Using only `used` portion of binary returns the same result

    let tuple = result.unwrap();
    let used_term = erlang::element_2(tuple, 2.into_process(&mut process), &mut process).unwrap();
    let used: usize = used_term.try_into_in_process(&mut process).unwrap();

    let prefix_term = Term::subbinary(binary_term, 0, 0, used, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_2(prefix_term, options, &mut process),
        Ok(tuple),
        process
    );

    // Without used returns only term

    assert_eq_in_process!(
        erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
        Ok(term),
        process
    );
}
