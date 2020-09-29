-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (List) when is_list(List) -> ignore;
    (Term) -> test(Term)
  end).

test(Left) ->
  Right = [],
  true = is_list(Right),
  test:caught(fun () ->
    Left ++ Right
  end).
