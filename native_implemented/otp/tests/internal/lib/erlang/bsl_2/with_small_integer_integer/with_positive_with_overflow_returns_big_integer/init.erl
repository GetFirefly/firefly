-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Integer = 2#1,
  true = is_small_integer(Integer),
  Shift = 64,
  true = (Shift > 0),
  Final = Integer bsl Shift,
  display(is_big_integer(Final)),
  display(Final).
