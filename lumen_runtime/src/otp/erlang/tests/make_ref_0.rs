use super::*;

#[test]
fn returns_a_unique_reference() {
    with_process(|process| {
        let first_reference = erlang::make_ref_0(&process);
        let second_reference = erlang::make_ref_0(&process);

        assert_eq!(first_reference, first_reference);
        assert_ne!(first_reference, second_reference);
        assert_eq!(second_reference, second_reference);
    })
}
