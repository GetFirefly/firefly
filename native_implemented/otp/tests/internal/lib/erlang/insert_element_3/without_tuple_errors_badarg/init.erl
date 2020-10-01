-module(init).
-export([start/0]).
-import(erlang, [insert_element/3]).

start() ->
  test:each(fun
    (Tuple) when is_tuple(Tuple) -> ignore;
    (Term) -> test(Term)
  end).

test(Tuple) ->
  Index = 1,
  Term = 1,
  test:caught(fun () ->
    insert_element(Index, Tuple, Term)
  end).
