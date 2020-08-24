-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    ChildPid = spawn(fun () ->
      wait_to_shutdown(),
      exit(abnormal)
    end),
    display(link(ChildPid)),
    shutdown(ChildPid),
    receive
      Message -> display(Message)
    after
      10 ->
        display({child, alive, is_process_alive(ChildPid)})
    end
  end),
  receive
    %% FIXME https://github.com/lumen/lumen/issues/546
    {'DOWN', ParentMonitorReference, process, _, {exit, Reason}} ->
      display({parent, exited, Reason});
    10 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
