-module(init).
-export([start/0]).
-import(erlang, [demonitor/1, display/1, process_info/2, spawn_monitor/1]).

start() ->
  ParentPid = self(),
  {ChildPid, MonitorReference} = spawn_monitor(fun () ->
     receive
       next -> ParentPid ! child_done
     end
  end),
  display(has_message(ChildPid, MonitorReference)),
  ChildPid ! next,
  receive
    child_done -> ok
  end,
  receive
  after 5 ->
    ok
  end,
  display(has_message(ChildPid, MonitorReference)),
  display(demonitor(MonitorReference)),
  display(has_message(ChildPid, MonitorReference)).

has_message(ChildPid, MonitorReference) ->
  {messages, Messages} = process_info(self(), messages),
  has_message(ChildPid, MonitorReference, Messages).

has_message(_ChildPid, _MonitorReference, []) ->
  false;
has_message(ChildPid, MonitorReference, [{'DOWN', Monitor, process, ChildPid, normal} | _T]) ->
  true;
has_message(ChildPid, MonitorReference, [_H, T]) ->
  has_message(ChildPid, MonitorReference, T).
