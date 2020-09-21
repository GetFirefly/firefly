-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).

start() ->
  from_fun(),
  from_arguments().

from_fun() ->
  Fun = fun from_fun_export/1,
  Return = apply(Fun, [ignored_argument]),
  display(Return).

from_fun_export(_) ->
  from_fun.

from_arguments() ->
  Fun = fun from_arguments_export/2,
  Return = apply(Fun, [argument_a, argument_b]),
  display(Return).

from_arguments_export(A, B) ->
  [A, B].
