-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Start) ->
  Length = 0,
  test:caught(fun () ->
    binary_part(<<>>, Start, Length)
  end).
