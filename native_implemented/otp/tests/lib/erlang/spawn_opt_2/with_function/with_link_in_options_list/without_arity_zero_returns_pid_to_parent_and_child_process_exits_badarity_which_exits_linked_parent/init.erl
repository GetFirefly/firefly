-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  lumen:log_exit(false),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    Options = [link],
    ChildPid = spawn_opt(fun (_) ->
      wait_to_shutdown()
    end, Options),
    ChildMonitorRef = monitor(process, ChildPid),
    receive
      {'DOWN', ChildMonitorRef, process, _, _} ->
        ok
    after
      10 ->
        ok
    end,
    display({child, alive, is_process_alive(ChildPid)})
  end),
  receive
    {'DOWN', ParentMonitorReference, process, _, Info} ->
      case Info of
        %% FIXME https://github.com/lumen/lumen/issues/548
        {badarity = Reason, _FunArgs} ->
          display({parent, Reason});
        _ ->
          display({parent, Info})
      end
  after
    10 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end,
  ok.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
