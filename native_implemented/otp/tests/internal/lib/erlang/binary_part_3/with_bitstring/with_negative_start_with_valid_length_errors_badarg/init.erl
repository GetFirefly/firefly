-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Start = -1,
  display(Start < 0),
  Length = 0,
  test:caught(fun () ->
    binary_part(<<>>, Start, Length)
  end).
