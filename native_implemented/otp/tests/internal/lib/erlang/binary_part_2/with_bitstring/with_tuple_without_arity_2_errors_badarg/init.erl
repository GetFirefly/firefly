-module(init).
-export([start/0]).
-import(erlang, [make_tuple/2]).

start() ->
  test(0),
  test(1),
  test(3).

test(TupleSize) ->
  StartLength = make_tuple(TupleSize, 0),
  test:caught(fun () ->
    binary_part(<<>>, StartLength)
  end).
