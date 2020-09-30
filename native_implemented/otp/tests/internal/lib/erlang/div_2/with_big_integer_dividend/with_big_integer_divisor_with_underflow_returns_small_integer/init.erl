-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  Dividend = test:big_integer(),
  Divisor = Dividend,
  true = is_big_integer(Divisor),
  Quotient = Dividend div Divisor,
  display(is_small_integer(Quotient)),
  display(Quotient).
