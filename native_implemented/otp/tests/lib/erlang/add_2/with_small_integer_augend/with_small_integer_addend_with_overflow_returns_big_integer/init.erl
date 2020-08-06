-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Sum = augend() + addend(),
  display(is_big_integer(Sum)).

augend() ->
  SmallInteger = (1 bsl 45),
  display(is_small_integer(SmallInteger)),
  SmallInteger.

addend() ->
  SmallInteger = (1 bsl 45),
  display(is_small_integer(SmallInteger)),
  SmallInteger.
