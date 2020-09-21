-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  Sum = augend() + addend(),
  display(is_float(Sum)).

augend() ->
  BigInteger = (1 bsl 46),
  display(is_big_integer(BigInteger)),
  BigInteger.

addend() ->
  3.0.


