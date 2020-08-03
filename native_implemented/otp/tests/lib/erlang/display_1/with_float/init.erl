-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(1.2),
  display(0.3),
  display(-4.5).
