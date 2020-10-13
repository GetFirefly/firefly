-module(init).
-export([start/0]).
-import(erlang, [seq_trace_info/1]).

start() ->
  test:caught(fun () ->
    seq_trace_info(invalid_atom)
  end).
