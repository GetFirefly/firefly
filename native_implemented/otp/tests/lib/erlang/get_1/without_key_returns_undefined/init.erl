-module(init).
-export([start/0]).
-import(erlang, [display/1, eget/1]).

start() ->
  Value = get(key),
  display(Value).

