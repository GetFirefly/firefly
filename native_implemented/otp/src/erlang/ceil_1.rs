#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

number_to_integer!(ceil);
