-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> test(Integer);
    (_) -> ignore
  end).

test(Dividend) ->
  Divisor = 0,
  test:caught(fun () ->
    Dividend div Divisor
  end).
