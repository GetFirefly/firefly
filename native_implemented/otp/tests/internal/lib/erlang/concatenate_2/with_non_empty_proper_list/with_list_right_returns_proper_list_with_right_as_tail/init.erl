-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Left = [1, 2],
  display(Left ++ []),
  display(Left ++ [3]),
  display(Left ++ [3, 4]).
