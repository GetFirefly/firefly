-module(init).
-export([start/0]).
-import(erlang, [display/1, integer_to_list/1]).

start() ->
  display(integer_to_list(-1)),
  display(integer_to_list(0)),
  display(integer_to_list(1)).

