-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Module = init,
  Function = child,
  Args = [one, two],
  {ChildPid, ChildMonitorReference} = spawn_monitor(Module, Function, Args),
  receive
    {'DOWN', ChildMonitorReference, process, _, Info} ->
      display({child, exited, Info}),
      display({parent, alive, true})
  after
    10 ->
      display({child, alive, is_process_alive(ChildPid)})
  end.

child(A, B) when is_integer(A) and is_integer(B) ->
  display({child, sum, A + B}).

