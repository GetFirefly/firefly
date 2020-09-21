-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Options = [],
  ChildPid = spawn_opt(fun () ->
    display(from_fun),
    wait_to_shutdown()
  end, Options),
  shutdown(ChildPid).

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
