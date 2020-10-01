-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (List) when is_list(List) -> ignore;
    (Term) -> test(Term)
  end).

test(List) ->
  test:caught(fun () ->
    hd(List)
  end).
