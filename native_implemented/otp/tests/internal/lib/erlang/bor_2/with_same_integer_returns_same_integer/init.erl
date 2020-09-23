-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> test(Integer);
    (_) -> ignore
  end).

test(Operand) ->
  Left = Operand,
  Right = Operand,
  Final = Left bor Right,
  display(Operand == Final).
