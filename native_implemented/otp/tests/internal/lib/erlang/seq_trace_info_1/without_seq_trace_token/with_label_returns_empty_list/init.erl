-module(init).
-export([start/0]).
-import(erlang, [display/1, seq_trace_info/1]).

start() ->
  display(seq_trace_info(label)).
