-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Boolean) when is_boolean(Boolean) -> ignore;
    (Term) -> caught(Term)
  end).

caught(Term) ->
  test:caught(fun () ->
     Term and true
  end).
