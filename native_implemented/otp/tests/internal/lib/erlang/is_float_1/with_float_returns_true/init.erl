-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(is_float(test:float())).
