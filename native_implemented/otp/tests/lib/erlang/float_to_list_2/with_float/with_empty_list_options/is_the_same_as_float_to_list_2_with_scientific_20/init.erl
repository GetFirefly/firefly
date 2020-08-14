-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_list/2]).

start() ->
  Options = [],
  display(float_to_list(0.0, Options)),
  display(float_to_list(0.1, Options)).

