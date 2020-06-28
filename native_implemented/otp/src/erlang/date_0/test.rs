use crate::erlang::date_0;
use crate::erlang::tuple_size_1;
use crate::test::with_process;

#[test]
fn returns_date_as_a_three_tuple() {
    with_process(|process| {
        let date_result = date_0::result(process).unwrap();
        let tuple_size_result = tuple_size_1::result(process, date_result).unwrap();

        assert_eq!(tuple_size_result, process.integer(3).unwrap());
    });
}
