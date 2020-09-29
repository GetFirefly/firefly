-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Reference) when is_reference(Reference) -> ignore;
    (Term) -> test(Term)
  end).

test(Reference) ->
  test:caught(fun () ->
    demonitor(Reference)
  end).
