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
  ChildPid ! next,
  receive
  after 5 ->
    ok
  end,
  display(has_message(MonitorReference)),
  Options = [flush, info],
  display(demonitor(MonitorReference, Options)),
  display(has_message(MonitorReference)).

has_message(MonitorReference) ->
  {messages, Messages} = process_info(self(), messages),
  has_message(MonitorReference, Messages).

has_message(_MonitorReference, []) ->
  false;
has_message(MonitorReference, [H | T]) ->
  case H of
    {'DOWN', MonitorReference, process, _ChildPid, normal} -> true;
    _ -> has_message(MonitorReference, T)
  end.
