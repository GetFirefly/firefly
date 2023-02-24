-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
    display(-1),
    display(-2),
    display(-3 + -4),
    adder(-3, -4).

adder(X, Y) ->
    display(X + Y).
