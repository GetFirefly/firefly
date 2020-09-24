-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> test(Integer);
    (_) -> ignore
  end).

test(Integer) ->
  Shift = 0,
  Final = Integer bsr Shift,
  display(Final == Integer).
