-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
  Args = [],
  Options = [],
  spawn_opt(Module, Function, Args, Options),
  wait_to_shutdown(),
  display({parent, alive}).

child() ->
  display({in, child}),
  exit(abnormal).

wait_to_shutdown() ->
  receive
    shutdown -> ok
  after
    10 -> ok
  end.
