-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Left) ->
  Right = 0,
  true = is_integer(Right),
  test:caught(fun () ->
    Left bor Right
  end).
