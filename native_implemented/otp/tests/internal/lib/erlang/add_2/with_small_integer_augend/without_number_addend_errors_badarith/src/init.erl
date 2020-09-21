-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term) -> caught(Term)
  end).

caught(Addend) ->
  test:caught(fun () ->
    augend() + Addend
  end).

augend() ->
  test:small_integer().
