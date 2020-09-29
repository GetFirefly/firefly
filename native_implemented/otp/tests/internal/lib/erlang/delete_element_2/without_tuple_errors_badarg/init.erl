-module(init).
-export([start/0]).
-import(erlang, [delete_element/2]).

start() ->
  test:each(fun
    (Tuple) when is_tuple(Tuple) -> ignore;
    (Term) -> test(Term)
  end).

test(Tuple) ->
  test:caught(fun () ->
    Index = 1,
    delete_element(Index, Tuple)
  end).
