-module(init).
-export([start/0]).
-import(erlang, [seq_trace/2]).

start() ->
  test:caught(fun () ->
    seq_trace(unsupported_flag, false)
  end).
