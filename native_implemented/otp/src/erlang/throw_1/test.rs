use crate::erlang::throw_1::result;
use crate::test::strategy;

#[test]
fn throws_reason() {
    run!(
        |arc_process| strategy::term(arc_process.clone()),
        |reason| {
            let actual = result(reason);

            if let Err(liblumen_alloc::erts::exception::Exception::Runtime(
                liblumen_alloc::erts::exception::RuntimeException::Throw(ref throw),
            )) = actual
            {
                let source_message = format!("{:?}", throw.source());
                let expected_substring = "explicit throw from Erlang";

                assert!(
                    source_message.contains(expected_substring),
                    "source message ({}) does not contain {:?}",
                    source_message,
                    expected_substring
                );
            } else {
                panic!("expected to throw, but got {:?}", actual);
            }

            Ok(())
        },
    );
}
