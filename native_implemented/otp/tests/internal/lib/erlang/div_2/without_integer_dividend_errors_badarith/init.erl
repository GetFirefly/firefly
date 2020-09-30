-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Dividend) ->
  Divisor = 1,
  test:caught(fun () ->
    Dividend div Divisor
  end).
