-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
  Args = [],
  ChildPid = spawn(Module, Function, Args),
  ChildMonitorReference = monitor(process, ChildPid),
  receive
    {'DOWN', ChildMonitorReference, process, _, Info} ->
      display({child, exited, Info}),
      display({parent, alive, true})
  after
    10 ->
      display({child, alive, is_process_alive(ChildPid)})
  end.

child() ->
  display({child, ran}).
