-module(init).
-export([start/0]).
-import(erlang, [binary_to_term/1, display/1]).

start() ->
  display(binary_to_term(<<131, 70, 63, 240, 0, 0, 0, 0, 0, 0>>)).
