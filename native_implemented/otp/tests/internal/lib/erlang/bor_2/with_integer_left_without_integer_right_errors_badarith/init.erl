-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Right) ->
  Left = 0,
  true = is_integer(Left),
  test:caught(fun () ->
    Left bor Right
  end).
