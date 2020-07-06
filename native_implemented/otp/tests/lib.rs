#[macro_use]
#[path = "./test.rs"]
mod test;

// Linux currently can't compile `lumen` compiler
#[cfg(not(target_os = "linux"))]
#[path = "lib/erlang.rs"]
pub mod erlang;
