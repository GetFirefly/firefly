-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(1),
  display(0),
  display(-1).
