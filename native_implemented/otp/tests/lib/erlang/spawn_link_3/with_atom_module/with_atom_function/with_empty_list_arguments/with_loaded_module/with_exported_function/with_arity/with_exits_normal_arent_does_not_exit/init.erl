-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
  Args = [],
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    spawn_link(Module, Function, Args),
    %% wait for exit to propagate
    wait_to_shutdown()
  end),
  receive
    {'DOWN', ParentMonitorReference, process, _, Info} ->
      display({parent, exited, Info})
  after 10 ->
    display({parent, alive, is_process_alive(ParentPid)})
  end,
  shutdown(ParentPid).

child() ->
  display({in, child}).

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.

shutdown(Pid) ->
  Pid ! shutdown.
