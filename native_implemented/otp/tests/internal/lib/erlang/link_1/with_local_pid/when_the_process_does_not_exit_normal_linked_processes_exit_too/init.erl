-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  StartPid = self(),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    ChildPid = spawn(fun () ->
      wait_to_shutdown()
    end),
    display(link(ChildPid)),
    StartPid ! {child_pid, ChildPid},
    wait_to_shutdown(),
    exit(abnormal)
  end),
  receive
    {child_pid, ChildPid} ->
      ChildMonitorReference = monitor(process, ChildPid),
      shutdown(ParentPid),
      receive
        {'DOWN', ParentMonitorReference, process, _, Reason1} ->
          display({parent, exited, Reason1})
      after 10 ->
        display({parent, alive, is_process_alive(ParentPid)})
      end,
      receive
        {'DOWN', ChildMonitorReference, process, _, Reason2} ->
          display({child, exited, Reason2})
      after 10 ->
        display({child, alive, is_process_alive(ChildPid)})
      end
  end.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
