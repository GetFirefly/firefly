use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

mod with_safe {
    use super::*;

    #[test]
    fn with_binary_encoding_atom_that_does_not_exist_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary(:atom3)
        let binary_term =
            Term::slice_to_binary(&[131, 100, 0, 5, 97, 116, 111, 109, 51], &mut process);
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_badarg!(erlang::binary_to_term_2(binary_term, options, &mut process));

        assert_eq!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::str_to_atom("atom3", DoNotCare).unwrap())
        );
    }

    #[test]
    fn with_binary_encoding_list_containing_atom_that_does_not_exist_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary([:atom1])
        let binary_term = Term::slice_to_binary(
            &[131, 108, 0, 0, 0, 1, 100, 0, 5, 97, 116, 111, 109, 49, 106],
            &mut process,
        );
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_badarg!(erlang::binary_to_term_2(binary_term, options, &mut process));

        assert_eq!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::cons(
                Term::str_to_atom("atom1", DoNotCare).unwrap(),
                Term::EMPTY_LIST,
                &mut process
            ))
        );
    }

    #[test]
    fn with_binary_encoding_small_tuple_containing_atom_that_does_not_exist_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary({:atom2})
        let binary_term = Term::slice_to_binary(
            &[131, 104, 1, 100, 0, 5, 97, 116, 111, 109, 50],
            &mut process,
        );
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_badarg!(erlang::binary_to_term_2(binary_term, options, &mut process));

        assert_eq!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::slice_to_tuple(
                &[Term::str_to_atom("atom2", DoNotCare).unwrap()],
                &mut process
            ))
        );
    }

    #[test]
    fn with_binary_encoding_small_atom_utf8_that_does_not_exist_errors_badarg() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
        let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
        let mut process = process_rw_lock.write().unwrap();
        // :erlang.term_to_binary(:"😈1")
        let binary_term =
            Term::slice_to_binary(&[131, 119, 5, 240, 159, 152, 136, 49], &mut process);
        let options = Term::cons(
            Term::str_to_atom("safe", DoNotCare).unwrap(),
            Term::EMPTY_LIST,
            &mut process,
        );

        assert_badarg!(erlang::binary_to_term_2(binary_term, options, &mut process));

        assert_eq!(
            erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
            Ok(Term::str_to_atom("😈1", DoNotCare).unwrap())
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
        Term::str_to_atom("used", DoNotCare).unwrap(),
        Term::EMPTY_LIST,
        &mut process,
    );

    let term = Term::str_to_atom("hello", DoNotCare).unwrap();
    let result = erlang::binary_to_term_2(binary_term, options, &mut process);

    assert_eq!(
        result,
        Ok(Term::slice_to_tuple(
            &[term, 9.into_process(&mut process)],
            &mut process
        ))
    );

    // Using only `used` portion of binary returns the same result

    let tuple = result.unwrap();
    let used_term = erlang::element_2(tuple, 2.into_process(&mut process), &mut process).unwrap();
    let used: usize = used_term.try_into().unwrap();

    let prefix_term = Term::subbinary(binary_term, 0, 0, used, 0, &mut process);

    assert_eq!(
        erlang::binary_to_term_2(prefix_term, options, &mut process),
        Ok(tuple)
    );

    // Without used returns only term

    assert_eq!(
        erlang::binary_to_term_2(binary_term, Term::EMPTY_LIST, &mut process),
        Ok(term)
    );
}
