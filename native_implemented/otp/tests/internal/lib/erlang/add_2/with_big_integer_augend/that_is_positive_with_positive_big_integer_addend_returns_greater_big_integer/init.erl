-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  Augend = augend(),
  Sum = Augend + addend(),
  display(Sum > Augend),
  display(is_big_integer(Sum)).

augend() ->
  BigInteger = (1 bsl 46),
  display(is_big_integer(BigInteger)),
  display(BigInteger > 0),
  BigInteger.

addend() ->
  BigInteger = (1 bsl 46),
  display(is_big_integer(BigInteger)),
  display(BigInteger > 0),
  BigInteger.


