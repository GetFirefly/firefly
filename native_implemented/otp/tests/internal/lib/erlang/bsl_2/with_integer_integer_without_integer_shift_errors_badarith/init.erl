-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Shift) ->
  Integer = 0,
  true = is_integer(Integer),
  test:caught(fun () ->
    Integer bsl Shift
  end).
