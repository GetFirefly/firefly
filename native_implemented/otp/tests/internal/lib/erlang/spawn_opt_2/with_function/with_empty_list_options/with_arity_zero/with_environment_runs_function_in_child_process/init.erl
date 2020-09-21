-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Environment = environment(),
  Options = [],
  ChildPid = spawn_opt(fun () ->
    display(Environment),
    wait_to_shutdown()
  end, Options),
  shutdown(ChildPid).

environment() ->
  from_environment.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
