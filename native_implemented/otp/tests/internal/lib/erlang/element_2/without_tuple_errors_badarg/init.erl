-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Tuple) when is_tuple(Tuple) -> ignore;
    (Term) -> test(Term)
  end).

test(Tuple) ->
  N = 1,
  test:caught(fun () ->
    element(N, Tuple)
  end).
