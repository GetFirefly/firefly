-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(2.0 / 4.0).
