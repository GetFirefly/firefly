-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> test(Number);
    (_) -> ignore
  end).

test(Dividend) ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term) -> test(Dividend, Term)
  end).

test(Dividend, Divisor) ->
  test:caught(fun () ->
    Dividend / Divisor
  end).
