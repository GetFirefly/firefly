-module(init).
-export([start/0]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term)  -> caught(Term)
  end).

caught(Term) ->
  test:caught(fun () ->
    abs(Term)
  end).
