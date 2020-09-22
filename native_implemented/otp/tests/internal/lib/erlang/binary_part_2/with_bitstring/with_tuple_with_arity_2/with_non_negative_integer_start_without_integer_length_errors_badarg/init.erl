-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Length) ->
  Start = 0,
  StartLength = {Start, Length},
  test:caught(fun () ->
    binary_part(<<>>, StartLength)
  end).
