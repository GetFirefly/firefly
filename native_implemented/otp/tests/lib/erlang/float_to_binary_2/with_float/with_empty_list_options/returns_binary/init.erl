-module(init).
-export([start/0]).
-import(erlang, [display/1, float_to_binary/1]).

start() ->
  Options = [],
  display(float_to_binary(-1.2, Options)),
  display(float_to_binary(0.3, Options)),
  display(float_to_binary(4.5, Options)).

