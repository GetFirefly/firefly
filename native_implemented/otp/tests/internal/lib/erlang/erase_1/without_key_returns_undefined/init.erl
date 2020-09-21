-module(init).
-export([start/0]).
-import(erlang, [display/1, erase/1]).

start() ->
  Value = erase(key),
  display(Value).

