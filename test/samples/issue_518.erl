-module(init).
-export([start/0]).
-import(erlang, [binary_to_integer/2, display/1, integer_to_binary/2]).

start() ->
  duals().

duals() ->
  duals(2, 36).

duals(MaxBase, MaxBase) ->
  dual(MaxBase);
duals(Base, MaxBase) ->
  dual(Base),
  duals(Base + 1, MaxBase).

dual(Base) ->
  display(36 == binary_to_integer(integer_to_binary(36, Base), Base)).
