-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(binary:encode_unsigned(70368744177664)).
