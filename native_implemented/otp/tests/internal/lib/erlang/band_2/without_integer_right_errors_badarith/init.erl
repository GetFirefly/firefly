-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test_right(Term)
  end).

test_right(Right) ->
  test:caught(fun () ->
    1 band Right
  end).
