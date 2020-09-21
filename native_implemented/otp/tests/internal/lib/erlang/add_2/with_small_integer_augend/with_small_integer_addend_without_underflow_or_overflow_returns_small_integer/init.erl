-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Sum = augend() + addend(),
  display(is_small_integer(Sum)).

augend() ->
  SmallInteger = 2,
  display(is_small_integer(SmallInteger)),
  SmallInteger.

addend() ->
  SmallInteger = 3,
  display(is_small_integer(SmallInteger)),
  SmallInteger.
