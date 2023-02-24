-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display([[]]),
  display([1, 2]),
  display([[1, 2, [3, 4]]]).
