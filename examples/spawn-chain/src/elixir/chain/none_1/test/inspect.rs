use locate_code::locate_code;

use super::*;

#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let time_value = arc_process.stack_peek(1).unwrap();

    lumen_runtime::system::io::puts(&format!("{}", time_value));
    arc_process.remove_last_frame(1);

    Process::call_code(arc_process)
}
