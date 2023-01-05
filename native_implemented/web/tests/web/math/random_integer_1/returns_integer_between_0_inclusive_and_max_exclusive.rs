//! ```elixir
//! exclusive_max = 2
//! random_integer = Lumen.Web.Math.random_integer(exclusive_max)
//! ```

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::math::random_integer_1;

pub const EXCLUSIVE_MAX: usize = 2;

#[native_implemented::function(Elixir.Lumen.Web.Math.RandomInteger1:returns_integer_between_0_inclusive_and_max_exclusive/0)]
fn result(process: &Process) -> Term {
    let exclusive_max = process.integer(EXCLUSIVE_MAX);

    // ```elixir
    // # pushed to stack: (exclusive_max)
    // # returned from call: N/A
    // # full stack: ()
    // # returns: random_integer
    // ```
    process.queue_frame_with_arguments(
        random_integer_1::frame().with_arguments(false, &[exclusive_max]),
    );

    Term::NONE
}
