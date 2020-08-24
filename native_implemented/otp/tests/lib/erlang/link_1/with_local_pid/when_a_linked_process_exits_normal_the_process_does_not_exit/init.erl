-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  {ChildPid, ChildMonitorReference} = spawn_monitor(fun () ->
    wait_to_shutdown()
  end),
  display(link(ChildPid)),
  shutdown(ChildPid),
  receive
    {'DOWN', ChildMonitorReference, process, _, Reason} ->
      display({child, exited, Reason})
  after
    10 ->
      display({child, alive, is_process_alive(ChildPid)})
  end.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
