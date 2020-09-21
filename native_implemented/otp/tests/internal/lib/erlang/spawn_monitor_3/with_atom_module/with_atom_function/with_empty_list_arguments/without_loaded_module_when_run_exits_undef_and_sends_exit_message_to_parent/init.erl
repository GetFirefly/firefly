-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  %% Typo
  Module = erlan,
  Function = self,
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


