-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  StartPid = self(),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    ChildPid = spawn_link(fun () ->
      wait_to_shutdown(),
      exit(abnormal)
    end),
    StartPid ! {child_pid, ChildPid},
    process_flag(trap_exit, true),
    display(process_info(self(), trap_exit)),
    process_flag(trap_exit, false),
    display(process_info(self(), trap_exit)),
    wait_to_shutdown_child(),
    shutdown(ChildPid),
    wait_to_shutdown()
  end),
  receive
    {child_pid, ChildPid} ->
      ChildMonitorReference = monitor(process, ChildPid),
      shutdown_child(ParentPid),
      receive
        {'DOWN', ChildMonitorReference, process, _, Reason2} ->
          display({child, exited, Reason2})
      after 10 ->
        display({child, alive, is_process_alive(ChildPid)})
      end,
      receive
        {'DOWN', ParentMonitorReference, process, _, Reason1} ->
          display({parent, exited, Reason1})
        after 10 ->
          display({parent, alive, is_process_alive(ParentPid)})
      end,
      shutdown(ParentPid)
    after 10 ->
      display({parent, did, 'not', send, child_pid})
  end.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.

shutdown_child(Pid) ->
  Pid ! shutdown_child.

wait_to_shutdown_child() ->
  receive
    shutdown_child -> ok
  end.
