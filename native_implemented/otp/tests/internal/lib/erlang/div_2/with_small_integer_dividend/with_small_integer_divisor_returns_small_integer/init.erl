-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Dividend = 1 bsl 40,
  true = is_small_integer(Dividend),
  Divisor = 1 bsl 30,
  true = is_small_integer(Divisor),
  Final = Dividend div Divisor,
  display(is_small_integer(Final)),
  display(Final).
