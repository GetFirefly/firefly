-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Sum = augend() + addend(),
  IsFloat = is_float(Sum),
  display(IsFloat).

augend() ->
  0.1.

addend() ->
  1 bsl 63.
