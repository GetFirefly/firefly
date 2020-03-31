use bus::Bus;

use super::Signal;

// Signal handling doesn't apply to WebAssembly
pub fn init(_bus: Bus<Signal>) {}
