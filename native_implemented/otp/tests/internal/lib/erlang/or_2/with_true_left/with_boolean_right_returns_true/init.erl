-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(true or false),
  display(true or true).
