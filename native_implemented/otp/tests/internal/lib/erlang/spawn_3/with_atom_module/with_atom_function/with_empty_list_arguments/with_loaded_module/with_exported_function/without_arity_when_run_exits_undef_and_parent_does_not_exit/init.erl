-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = erlang,
  Function = '+',
  Args = [],
  ChildPid = spawn(Module, Function, Args),
  ChildMonitorReference = monitor(process, ChildPid),
  receive
    {'DOWN', ChildMonitorReference, process, _, _} ->
      display({parent, alive, true})
  after
    10 ->
      display({child, alive, is_process_alive(ChildPid)})
  end.


