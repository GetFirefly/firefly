-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Sum = augend() + addend(),
  display(Sum).

augend() ->
  2.0.

addend() ->
  3.0.
