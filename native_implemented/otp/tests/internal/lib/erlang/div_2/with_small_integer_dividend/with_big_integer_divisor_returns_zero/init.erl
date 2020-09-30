-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Dividend = 1 bsl 40,
  true = is_small_integer(Dividend),
  Divisor = test:big_integer(),
  Final = Dividend div Divisor,
  display(is_big_integer(Final)),
  display(Final).
