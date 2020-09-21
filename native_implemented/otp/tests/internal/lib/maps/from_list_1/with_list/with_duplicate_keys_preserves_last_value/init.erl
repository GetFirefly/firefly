-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(maps:from_list([{key, first_value}, {key, last_value}])).
