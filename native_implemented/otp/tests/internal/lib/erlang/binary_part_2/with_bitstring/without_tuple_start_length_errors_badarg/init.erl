-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Tuple) when is_tuple(Tuple) -> ignore;
    (Term) -> test(Term)
  end).

test(StartLength) ->
  test:caught(fun () ->
    binary_part(<<>>, StartLength)
  end).
