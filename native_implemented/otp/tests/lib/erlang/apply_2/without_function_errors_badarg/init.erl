-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Function) when is_function(Function) -> ignore;
    (Term) -> caught(Term)
  end).

caught(Term) ->
  test:caught(fun () ->
    apply(Term, [])
  end).
