-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Integer = 2#10,
  true = is_small_integer(Integer),
  Shift = 3,
  true = (Shift > 0),
  Final = Integer bsr Shift,
  display(Final).
