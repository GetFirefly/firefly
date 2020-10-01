-module(init).
-export([start/0]).
-import(erlang, [insert_element/3]).

start() ->
  test({}, 0),
  test({1}, 0),
  test({1}, 3).

test(Tuple, Index) ->
  Element = inserted_element,
  test:caught(fun () ->
    insert_element(Index, Tuple, Element)
  end).
