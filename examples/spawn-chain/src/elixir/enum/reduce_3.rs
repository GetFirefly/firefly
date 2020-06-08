use anyhow::*;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::r#enum::reduce_range_dec_4;
use crate::elixir::r#enum::reduce_range_inc_4;

// Private

#[native_implemented::function(reduce/3)]
fn result(
    process: &Process,
    enumerable: Term,
    initial: Term,
    reducer: Term,
) -> exception::Result<Term> {
    match enumerable.decode().unwrap() {
        TypedTerm::Map(map) => {
            match map.get(atom!("__struct__")) {
                Some(struct_name) => {
                    if struct_name == atom!("Elixir.Range") {
                        // This assumes no one was cheeky and messed with the map
                        // representation of the struct
                        let first_key = atom!("first");
                        let first = map.get(first_key).unwrap();

                        let last_key = atom!("last");
                        let last = map.get(last_key).unwrap();

                        if first <= last {
                            process.queue_frame_with_arguments(
                                reduce_range_inc_4::frame()
                                    .with_arguments(false, &[first, last, initial, reducer]),
                            );
                        } else {
                            process.queue_frame_with_arguments(
                                reduce_range_dec_4::frame()
                                    .with_arguments(false, &[first, last, initial, reducer]),
                            );
                        }

                        Ok(Term::NONE)
                    } else {
                        Err(
                            anyhow!("enumerable ({}) is a struct, but not a Range", enumerable)
                                .into(),
                        )
                    }
                }
                None => {
                    Err(anyhow!("enumerable ({}) is a map, but not a struct", enumerable).into())
                }
            }
        }
        _ => Err(anyhow!("enumerable ({}) is not a map", enumerable).into()),
    }
}
