-module(init).
-export([start/0]).
-import(erlang, [ceil/1, display/1]).

start() ->
  display(ceil(-1.9)),
  display(ceil(-1.1)),
  display(ceil(-0.5)),
  display(ceil(0.5)),
  display(ceil(1.0)).
