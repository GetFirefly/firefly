-module(init).
-export([start/0]).
-import(erlang, [demonitor/2, display/1, process_info/2, spawn_monitor/1]).

start() ->
  ParentPid = self(),
  {ChildPid, MonitorReference} = spawn_monitor(fun () ->
    receive
      next -> ParentPid ! child_done
    end
  end),
  Options = [flush],
  display(demonitor(MonitorReference, Options)),
  ChildPid ! next.
