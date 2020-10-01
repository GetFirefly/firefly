-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test(test:binary()).

test(Binary) ->
  display(is_binary(Binary)).
