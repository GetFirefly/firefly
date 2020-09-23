-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  Integer = 2#101100111000111100001111100000111111000000111111100000001111111100000000,
  Shift = -1,
  Shifted = Integer bsl Shift,
  display(is_big_integer(Shifted)),
  display(Shifted).
