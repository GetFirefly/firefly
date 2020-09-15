-module(init).
-export([start/0]).
-import(erlang, [append_element/2]).

start() ->
  test:each(fun
    (Tuple) when is_tuple(Tuple) -> ignore;
    (Term) -> caught(Term)
  end).

caught(Term) ->
  test:caught(fun () ->
     append_element(Term, element)
  end).
