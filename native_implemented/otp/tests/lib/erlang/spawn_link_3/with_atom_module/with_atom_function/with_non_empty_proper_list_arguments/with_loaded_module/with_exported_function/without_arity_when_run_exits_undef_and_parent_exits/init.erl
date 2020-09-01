-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  Function = '+',
  Args = [1, 2, 2],
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
  end.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
