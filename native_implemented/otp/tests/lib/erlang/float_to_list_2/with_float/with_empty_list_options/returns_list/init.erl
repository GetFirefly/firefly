-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_list/1]).

start() ->
  Options = [],
  display(float_to_list(-1.2, Options)),
  display(float_to_list(0.3, Options)),
  display(float_to_list(4.5, Options)).

