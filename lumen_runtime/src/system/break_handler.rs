cfg_if::cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
     mod wasm32;

     pub use wasm32::*;
  } else if #[cfg(windows)] {
     mod windows;

     pub use windows::*;
  } else {
     mod std;

     pub use self::std::*;
  }
}

#[derive(Clone)]
pub enum Signal {
    Unknown,
    INT,
    TERM,
    QUIT,
    HUP,
    ABRT,
    ALRM,
    USR1,
    USR2,
    CHLD,
}
