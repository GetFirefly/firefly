-module(init).
-export([start/0]).
-import(erlang, [binary_to_term/1, display/1]).

start() ->
  display(binary_to_term(<<131, 109, 0, 0, 0, 2, 0, 1>>)).
