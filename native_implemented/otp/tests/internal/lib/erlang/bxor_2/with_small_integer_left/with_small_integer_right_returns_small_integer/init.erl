-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Left = 2#1100,
  true = is_small_integer(Left),
  Right = 2#1010,
  true = is_small_integer(Right),
  Final = Left bxor Right,
  display(is_small_integer(Final)),
  display(Final).
