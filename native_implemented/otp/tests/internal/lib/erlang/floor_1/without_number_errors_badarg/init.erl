-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term) -> test(Term)
  end).

test(Number) ->
  test:caught(fun () ->
    floor(Number)
  end).
