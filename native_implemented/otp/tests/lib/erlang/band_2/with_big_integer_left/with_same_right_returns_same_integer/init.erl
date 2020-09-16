-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Left = 2#1100110011001100110011001100110011001100110011001100110011001100,
  true = is_big_integer(Left),
  Right = Left,
  true = is_big_integer(Right),
  Final = Left band Right,
  display(Final == Left).
