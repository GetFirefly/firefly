-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Length = 0,
  Start = 1,
  test:caught(fun () ->
    binary_part(<<>>, Start, Length)
  end).
