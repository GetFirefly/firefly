-module(init).
-export([start/0]).
-import(erlang, [delete_element/2]).

start() ->
  test(-1),
  test(0),
  test(2).

test(Index) ->
  test:caught(fun () ->
    Tuple = {1},
    delete_element(Index, Tuple)
  end).
