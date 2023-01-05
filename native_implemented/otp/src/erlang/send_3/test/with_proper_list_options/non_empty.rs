use super::*;

mod with_noconnect;
mod with_noconnect_and_nosuspend;
mod with_nosuspend;

#[test]
fn with_invalid_option_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()).prop_filter(
                    "Option must be invalid",
                    |option| match option {
                        Term::Atom(atom) => match atom.as_str() {
                            "noconnect" | "nosuspend" => false,
                            _ => true,
                        },
                        _ => true,
                    },
                ),
            )
        },
        |(arc_process, message, option)| {
            let destination = arc_process.pid_term().unwrap();
            let options = arc_process.list_from_slice(&[option]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, destination, message, options),
                "supported options are noconnect or nosuspend"
            );

            Ok(())
        },
    );
}
