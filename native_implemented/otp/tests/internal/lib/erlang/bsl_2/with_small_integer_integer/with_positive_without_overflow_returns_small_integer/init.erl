-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Integer = 2#101100111000,
  true = is_small_integer(Integer),
  display(Integer > 0),
  Shift = -9,
  display(Integer bsl Shift).
