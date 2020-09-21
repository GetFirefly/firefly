-module(init).
-export([start/0]).
-import(erlang, [binary_to_integer/2, display/1]).

start() ->
  bases().

bases() ->
  bases(2, 36).

bases(Final, Final) ->
  base(Final);
bases(Base, Final) ->
  base(Base),
  bases(Base + 1, Final).

base(Base) ->
  display(binary_to_integer(<<"10">>, Base)).
