-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Sum = augend() + addend(),
  display(is_float(Sum)).

augend() ->
  2.

addend() ->
  3.0.
