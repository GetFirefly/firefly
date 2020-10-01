-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> test(Integer);
    (_) -> ignore
  end).

test(Number) ->
  Final = floor(Number),
  display(is_integer(Final)),
  display(Final =:= Number).
