-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  BigInteger = 2#1010101010101010101010101010101010101010101010101010101010101010,
  display(is_big_integer(BigInteger)),
  Final = bnot BigInteger,
  display(is_big_integer(Final)),
  display(Final).
