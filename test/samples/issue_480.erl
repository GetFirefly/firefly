-module(init).
-export([start/0]).
-import(erlang, [binary_to_float/1, display/1]).

start() ->
  display(binary_to_float(<<"-1.2">>)),
  display(binary_to_float(<<"0.0">>)),
  display(binary_to_float(<<"3.4">>)).
