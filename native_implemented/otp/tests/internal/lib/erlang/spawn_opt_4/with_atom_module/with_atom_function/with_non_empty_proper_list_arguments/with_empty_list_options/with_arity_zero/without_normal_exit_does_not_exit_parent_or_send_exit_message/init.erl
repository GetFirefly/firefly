-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
  Args = [1, 2],
  Options = [],
  spawn_opt(Module, Function, Args, Options),
  wait_to_shutdown(),
  display({parent, alive}).

child(A, B) ->
  display({in, child, A, B}),
  exit(abnormal).

wait_to_shutdown() ->
  receive
    shutdown -> ok
  after
    10 -> ok
  end.
