-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    Options = [link, monitor],
    spawn_opt(fun (_) ->
      ok
    end, Options),
    wait_to_shutdown()
  end),
  receive
    {'DOWN', ParentMonitorReference, process, _, Info} ->
      case Info of
        %% FIXME https://github.com/lumen/lumen/issues/548
        {Reason = badarity, _FunArgs} ->
          display({parent, Reason});
        _ ->
          display({parent, Info})
      end
  after
    10 ->
      display({parent, alive, is_process_alive(ParentPid)})
  end.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
