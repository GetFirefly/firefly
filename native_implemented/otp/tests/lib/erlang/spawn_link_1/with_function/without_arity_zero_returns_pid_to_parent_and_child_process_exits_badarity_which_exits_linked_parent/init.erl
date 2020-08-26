-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  lumen:log_exit(false),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    ChildPid = spawn_link(fun (_) ->
      wait_to_shutdown(),
      exit(shutdown)
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
    ok
  end),
  receive
    {'DOWN', ParentMonitorReference, process, _, Info} ->
      case Info of
        {badarity = Reason, FunArgs} ->
          display({parent, Reason});
        _ ->
          display({parent, Info})
      end
  after
    100 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end,
  ok.

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
