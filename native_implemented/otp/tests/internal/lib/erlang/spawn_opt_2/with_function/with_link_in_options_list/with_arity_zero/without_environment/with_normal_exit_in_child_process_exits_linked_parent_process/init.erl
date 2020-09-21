-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  lumen:log_exit(false),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    Options = [link],
    ChildPid = spawn_opt(fun () ->
      wait_to_shutdown()
    end, Options),
    ChildMonitorRef = monitor(process, ChildPid),
    shutdown(ChildPid),
    receive
      {'DOWN', ChildMonitorRef, process, _, Info} ->
        display({child, exited, Info})
    after
      10 ->
        display({child, alive, is_process_alive(ChildPid)})
    end,
    wait_to_shutdown()
  end),
  receive
    %% FIXME https://github.com/lumen/lumen/issues/546
    {'DOWN', ParentMonitorReference, process, _, Reason} ->
      display({parent, Reason})
  after
    10 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end,
  shutdown(ParentPid).

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
