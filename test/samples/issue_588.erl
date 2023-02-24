-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Left = 100,
  Right = float(Left),
  display(Left =:= Right),
  display(Left == Right).
