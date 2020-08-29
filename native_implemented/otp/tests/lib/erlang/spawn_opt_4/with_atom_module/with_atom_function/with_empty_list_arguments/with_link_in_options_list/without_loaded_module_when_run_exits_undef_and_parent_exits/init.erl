-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
    %% Typo
    Module = erlan,
    Function = '+',
    Args = [],
    Options = [link],
    spawn_opt(Module, Function, Args, Options),
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
  after
    10 -> ok
  end.
