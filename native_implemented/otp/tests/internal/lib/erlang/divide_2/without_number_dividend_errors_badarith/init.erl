-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term) -> test(Term)
  end).

test(Dividend) ->
  Divisor = 1,
  true = is_number(Divisor),
  test:caught(fun () ->
    Dividend / Divisor
  end).
