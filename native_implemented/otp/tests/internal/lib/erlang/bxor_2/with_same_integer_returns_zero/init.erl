-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> test(Integer);
    (_) -> ignore
  end).

test(Operand) ->
  display(Operand bxor Operand).
