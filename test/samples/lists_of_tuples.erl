-module(init).
-export([start/0]).

start() ->
  Tuple = [{async, false}],
  erlang:print(Tuple).
