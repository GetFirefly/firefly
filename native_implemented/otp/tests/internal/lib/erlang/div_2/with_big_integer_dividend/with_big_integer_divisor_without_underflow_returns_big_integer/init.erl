-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Dividend = 1 bsl 92,
  true = is_big_integer(Dividend),
  Divisor = 1 bsl 46,
  true = is_big_integer(Divisor),
  Quotient = Dividend div Divisor,
  display(is_big_integer(Quotient)),
  display(Quotient).
