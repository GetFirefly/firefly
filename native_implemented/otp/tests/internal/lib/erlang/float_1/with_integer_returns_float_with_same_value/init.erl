-module(init).
-export([start/0]).
-import(erlang, [display/1, float/1]).

start() ->
  display(float(-1)),
  display(float(0)),
  display(float(1)).

