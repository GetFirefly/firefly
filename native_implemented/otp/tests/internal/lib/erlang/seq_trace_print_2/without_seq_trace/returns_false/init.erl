-module(init).
-export([start/0]).
-import(erlang, [display/1, seq_trace_print/2]).

start() ->
  Label = label,
  Message = message,
  display(seq_trace_print(Label, Message)).
