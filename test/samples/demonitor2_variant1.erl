-module(init).
-export([start/0]).
-import(erlang, [demonitor/2, display/1, process_info/2, spawn_monitor/1]).

start() ->
  {ChildPid, MonitorReference} = spawn_monitor(fun () ->
     receive
       next -> ok
     end
  end),
  MonitorCountBefore = monitor_count(self()),
  MonitoredByCountBefore = monitored_by_count(ChildPid),
  Options = [],
  display(demonitor(MonitorReference, Options)),
  MonitorCountAfter = monitor_count(self()),
  MonitoredByCountAfter = monitored_by_count(ChildPid),
  display(MonitorCountBefore - 1 == MonitorCountAfter),
  display(MonitoredByCountBefore - 1 == MonitoredByCountAfter),
  ChildPid ! next.

monitor_count(Pid) ->
   {monitors, Monitors} = process_info(Pid, monitors),
   length(Monitors).

monitored_by_count(Pid) ->
  {monitored_by, MonitoredBys} = process_info(Pid, monitored_by),
  length(MonitoredBys).
