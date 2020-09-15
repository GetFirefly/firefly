-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (List) when is_list(List) -> ignore;
    (Term) -> caught(Term)
  end).

caught(Term) ->
  test:caught(fun () ->
    apply(fun () -> ok end, Term)
  end).
