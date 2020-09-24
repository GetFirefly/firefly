-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Start = 1,
  Length = -2,
  display(Start + Length < 0),
  test:caught(fun () ->
    binary_part(<<0, 1>>, Start, Length)
  end).
