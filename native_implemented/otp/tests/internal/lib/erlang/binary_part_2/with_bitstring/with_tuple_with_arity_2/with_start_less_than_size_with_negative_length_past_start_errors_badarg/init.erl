-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Start = 1,
  Length = -2,
  StartLength = {Start, Length},
  test:caught(fun () ->
    binary_part(<<0, 1>>, StartLength)
  end).
