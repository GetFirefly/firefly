use std::sync::OnceLock;

use parking_lot::RwLock;

use firefly_session::Input;

static INTERNED_INPUTS: OnceLock<RwLock<Vec<Input>>> = OnceLock::new();

pub type InputId = u32;

pub fn intern(input: Input) -> InputId {
    let inputs = INTERNED_INPUTS.get_or_init(|| RwLock::new(Vec::with_capacity(1024)));
    let mut guard = inputs.write();
    let id = guard.len() as InputId;
    guard.push(input);
    id
}

pub fn get(input: InputId) -> Input {
    let inputs = INTERNED_INPUTS.get().unwrap();
    let guard = inputs.read();
    guard[input].clone()
}
