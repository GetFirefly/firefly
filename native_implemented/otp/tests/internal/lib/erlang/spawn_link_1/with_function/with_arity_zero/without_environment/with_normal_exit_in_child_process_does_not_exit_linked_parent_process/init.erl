-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    ChildPid = spawn_link(fun () ->
      wait_to_shutdown()
    end),
    ChildMonitorRef = monitor(process, ChildPid),
    shutdown(ChildPid),
    receive
      {'DOWN', ChildMonitorRef, process, _, Info} ->
        display({child, exited, Info})
    after
      10 ->
        display({child, alive, is_process_alive(ChildPid)})
    end,
    wait_to_shutdown(),
    ok
                                                      end),
  receive
    {'DOWN', ParentMonitorReference, process, _, Info} ->
      display({parent, Info})
  after
    10 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end,
  shutdown(ParentPid),
  ok.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
