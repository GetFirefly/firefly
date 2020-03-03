import * as Interpreter from "interpreter-in-browser";

window.Interpreter = Interpreter;

function compile_url(url) {
  return fetch(url, {
    method: "GET"
  }).then(response => {
    if (!response.ok) {
      throw "fail";
    }
    return response.text().then(text => {
      Interpreter.compile_erlang_module(text);
      return text;
    });
  });
}

compile_url("ex_sample.erl").then(() => {
  let heap = new Interpreter.JsHeap(1000);
  let num = heap.integer(12);
  console.log("spawning...");
  heap.spawn("Elixir.ExSample", "run_me", [num], 100000);
});
