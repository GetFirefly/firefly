-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).

start() ->
  from_fun(),
  from_environment(),
  from_arguments(),
  from_environment_and_arguments().

from_fun() ->
  Fun = fun (_) ->
    from_fun
  end,
  Return = apply(Fun, [ignored_argument]),
  display(Return).

from_environment() ->
  A = a(),
  Fun = fun (_) ->
    A
  end,
  Return = apply(Fun, [ignored_argument]),
  display(Return).

from_arguments() ->
  Fun = fun (A, B) ->
    [A, B]
  end,
  Return = apply(Fun, [argument_a, argument_b]),
  display(Return).

from_environment_and_arguments() ->
  A = a(),
  Fun = fun (B, C) ->
    [A, B, C]
  end,
  Return = apply(Fun, [argument_a, argument_b]),
  display(Return).

a() ->
  from_environment.
