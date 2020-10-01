-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(test:big_integer()),
  test(test:small_integer()).

test(Integer) ->
  display(is_integer(Integer)).
