use bus::Bus;

use super::Signal;

// signal-hook says it supports Windows, but fails to build (https://cirrus-ci.com/task/5717029562089472)
pub fn init(_bus: Bus<Signal>) {}
