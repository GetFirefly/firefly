-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Left = [left],
  Right = [right_hd | right_tail],
  display(Left ++ Right).
