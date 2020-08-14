-module(init).
-export([start/0]).
-import(erlang, [display/1, get_keys/1]).

start() ->
  display(get_keys(value)).

