-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(test_big_integer:negative()),
  test(test_big_integer:positive()),
  test(test_float:negative()),
  test(test_float:zero()),
  test(test_float:positive()),
  test(test_small_integer:negative()),
  test(test_small_integer:zero()),
  test(test_small_integer:positive()).

test(Number) ->
  display(abs(Number)).
