-module(init).
-export([start/0]).

start() ->
  test(1, 2),
  test(1, 2.0),
  test(1.0, 2),
  test(1.0, 2.0).

test(Dividend, Divisor) ->
  test:caught(fun () ->
    Dividend / Divisor
  end).
