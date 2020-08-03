//! ```elixir
//! defmodule Chain do
//!   def counter(next_pid, output) do
//!     output.("spawned")
//!
//!     receive do
//!       n ->
//!         output.("received #{n}")
//!         sent = send next_pid, n + 1
//!         output.("sent #{sent} to #{next_pid}")
//!     end
//!   end
//!
//!   def create_processes(n, output) when is_function(output, 1) do
//!     last =
//!       Enum.reduce(
//!         1..n,
//!         self(),
//!         fn (_, send_to) ->
//!           spawn(Chain, :counter, [send_to, output])
//!         end
//!       )
//!
//!     send(last, 0) # start the count by sending a zero to the last process
//!
//!     receive do # and wait for the result to come back to us
//!       final_answer when is_integer(final_answer) ->
//!         "Result is #{inspect(final_answer)}"
//!          final_answer
//!     end
//!   end
//!
//!   def console(n) do
//!     run(n, &console_output/1)
//!   end
//!
//!   def dom(n) do
//!     run(n, &dom_output/1)
//!   end
//!
//!   def none(n) do
//!     run(n, &none_output/1)
//!   end
//!
//!   def on_submit(event) do
//!     {:ok, event_target} = Lumen.Web.Event.target(event)
//!     {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
//!     value_string = Lumen.Web.HTMLInputElement.value(n_input)
//!     n = :erlang.binary_to_integer(value_string)
//!     dom(n)
//!   end
//!
//!   # Private Functions
//!
//!   defp console_output(text) do
//!     IO.puts("#{self()} #{text}")
//!   end
//!
//!   defp dom_output(text) do
//!     window = Lumen.Web.Window.window()
//!     document = Lumen.Web.Window.document(window)
//!     {:ok, tr} = Lumen.Web.Document.create_element(document, "tr")
//!
//!     {:ok, pid_text} = Lumen.Web.Document.create_text_node(document, to_string(self()))
//!     {:ok, pid_td} = Lumen.Web.Document.create_element(document, "td")
//!     Lumen.Web.Element.append_child(pid_td, pid_text);
//!     Lumen.Web.Element.append_child(tr, pid_td)
//!
//!     {:ok, text_text} = Lumen.Web.Document.create_text_node(document, text)
//!     {:ok, text_td} = Lumen.Web.Document.create_element(document, "td")
//!     Lumen.Web.Element.append_child(text_td, text_text);
//!     Lumen.Web.Element.append_child(tr, text_td)
//!
//!     {:ok, output} = Lumen.Web.Document.get_element_by_id("output")
//!     Lumen.Web.Element.append_child(output, tr)
//!   end
//!
//!   defp none_output(_text) do
//!     :ok
//!   end
//!
//!   defp run(n, output) when is_function(output, 1) do
//!     {time, value} = :timer.tc(Chain, :create_processes, [n, output])
//!     output.("Chain.run(#{n}) in #{time} microseconds")
//!     {time, value}
//!   end
//! end
//! ```

pub mod console_1;
mod console_output_1;
pub mod counter_2;
pub mod create_processes_2;
mod create_processes_reducer_3;
pub mod dom_1;
mod dom_output_1;
pub mod none_1;
mod none_output_1;
pub mod on_submit_1;
mod run_2;

use liblumen_alloc::erts::term::prelude::Atom;

pub fn module() -> Atom {
    Atom::from_str("Elixir.Chain")
}

fn module_id() -> usize {
    module().id()
}
