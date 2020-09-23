-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  SmallInteger = 2#10,
  display(is_small_integer(SmallInteger)),
  Final = bnot SmallInteger,
  display(is_small_integer(Final)),
  display(Final).
