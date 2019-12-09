use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::throw_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn throws_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                let actual = native(reason);

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
            })
            .unwrap();
    });
}
