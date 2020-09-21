-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  %% One bit less than `Right`, so that only the small integer bits will be non-zero
  Left = 2#100000000000000000000000000000000000000000000000000000000001100,
  true = is_big_integer(Left),
  Right = 2#1000000000000000000000000000000000000000000000000000000000001010,
  true = is_big_integer(Right),
  Final = Left band Right,
  display(Final),
  display(is_small_integer(Final)).
