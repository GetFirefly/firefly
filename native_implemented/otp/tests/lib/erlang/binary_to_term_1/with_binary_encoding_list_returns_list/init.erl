-module(init).
-export([start/0]).
-import(erlang, [binary_to_term/1, display/1]).

start() ->
  display(binary_to_term(<<131, 108, 0, 0, 0, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1, 106>>)).
