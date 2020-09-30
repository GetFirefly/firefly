-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_small_integer/1]).

start() ->
  Dividend = test:big_integer(),
  Divisor = 1048576,
  true = is_small_integer(Divisor),
  Quotient = Dividend div Divisor,
  display(is_small_integer(Quotient)),
  display(Quotient).
