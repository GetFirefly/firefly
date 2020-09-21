-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
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

child() ->
  display({child, ran}).
