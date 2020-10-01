-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(-1.2),
  test(-1.0),
  test(-0.3),
  test(0.0),
  test(0.4),
  test(1.0),
  test(1.5).

test(Number) ->
  display({Number, floor(Number)}).
