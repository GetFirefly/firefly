-module(init).
-export([start/0]).
-import(erlang, [display/1, float/1]).

start() ->
  test:each(fun
    (Number) when is_number(Number) -> ignore;
    (Term) -> test(Term)
  end).

test(Number) ->
  test:caught(fun () ->
    float(Number)
  end).
