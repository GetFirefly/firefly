-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_binary/2]).

start() ->
  Options = [],
  display(float_to_binary(0.0, Options)),
  display(float_to_binary(0.1, Options)).

