-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(element(1, {1})),
  display(element(1, {1, 2})),
  display(element(2, {1, 2})).
