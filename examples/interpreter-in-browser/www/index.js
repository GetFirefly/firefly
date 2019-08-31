import * as Interpreter from "interpreter-in-browser";

window.Interpreter = Interpreter;

function compile_url(url) {
    return fetch(url, {
        method: "GET",
    }).then((response) => {
        if (!response.ok) {
            throw "fail";
        }
        return response.text()
            .then((text) => {
                Interpreter.compile_erlang_module(text);
                return text;
            });
    });
}

//compile_url("foo.erl")
//compile_url("chain/run.erl")
//    .then(() => compile_url("chain/timer.erl"))
//    .then(() => compile_url("chain/Elixir.Range.erl"))
//    .then(() => compile_url("chain/Elixir.Enum.erl"))
//    .then(() => compile_url("chain/Elixir.String.Chars.erl"))
//    .then(() => compile_url("chain/Elixir.Chain.erl"))
compile_url("dbmon/run.erl")
    .then(() => compile_url("dbmon/Elixir.TemplateDemo.Ast.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.Supervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.AppSupervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.WindowSupervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.DocumentSupervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.BodySupervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.ElementSupervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.DbMonDemo.TextWorker.erl"))
    .then(() => compile_url("dbmon/Elixir.Supervisor.erl"))
    .then(() => compile_url("dbmon/Elixir.GenServer.erl"))
    .then(() => compile_url("dbmon/Elixir.Keyword.erl"))
    .then(() => compile_url("dbmon/Elixir.Access.erl"))
    .then(() => compile_url("dbmon/Elixir.Enum.erl"))
    .then(() => compile_url("dbmon/Elixir.Process.erl"))
    .then(() => compile_url("dbmon/Elixir.List.erl"))
    .then(() => compile_url("dbmon/supervisor.erl"))
    .then(() => compile_url("dbmon/gen_server.erl"))
    .then(() => compile_url("dbmon/gen.erl"))
    .then(() => compile_url("dbmon/proc_lib.erl"))
    .then(() => compile_url("dbmon/lists.erl"))
    .then(() => {
        let heap = new Interpreter.JsHeap(1000);
        //let num = heap.integer(12);
        console.log("spawning...");
        heap.spawn("run", "run", [], 100000);
    });

