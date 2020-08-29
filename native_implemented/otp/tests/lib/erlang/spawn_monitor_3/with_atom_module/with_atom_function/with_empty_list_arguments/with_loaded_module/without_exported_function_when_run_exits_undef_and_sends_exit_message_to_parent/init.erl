-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  %% Typo
  Function = sel,
  Args = [],
  {ChildPid, ChildMonitorReference} = spawn_monitor(Module, Function, Args),
  receive
    {'DOWN', ChildMonitorReference, process, _, Info} ->
      display({child, exited, Info})
  after
    10 ->
      display({child, alive, is_process_alive(ChildPid)})
  end,
  display({parent, alive, true}).


