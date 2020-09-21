-module(init).
-export([start/0]).
-import(erlang, [display/1, float/1]).

start() ->
  display(float(-1.2)),
  display(float(0.3)),
  display(float(4.5)).

