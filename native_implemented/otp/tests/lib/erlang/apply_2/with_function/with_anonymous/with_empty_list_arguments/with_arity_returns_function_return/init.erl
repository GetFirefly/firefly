-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).

start() ->
  from_fun(),
  from_environment().

from_fun() ->
  Fun = fun () ->
    from_fun
  end,
  Return = apply(Fun, []),
  display(Return).

from_environment() ->
  A = from_environment,
  Fun = fun () ->
    A
  end,
  Return = apply(Fun, []),
  display(Return).
