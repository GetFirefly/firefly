-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_binary/1]).

start() ->
  display(float_to_binary(-1.2)),
  display(float_to_binary(0.3)),
  display(float_to_binary(4.5)).

