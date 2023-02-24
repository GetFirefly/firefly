-module(init).
-export([start/0]).
-import(erlang, [demonitor/1, display/1, process_info/2, spawn_monitor/1]).

start() ->
  {ChildPid, MonitorReference} = spawn_monitor(fun () ->
    ok
  end),
  wait(2),
  display(has_message({'DOWN', MonitorReference, process, ChildPid, normal})),
  display(demonitor(MonitorReference)),
  display(has_message({'DOWN', MonitorReference, process, ChildPid, normal})).

has_message(Message) ->
  Messages = process_info(self(), messages),
  lists:member(Message, Messages).

wait(Milliseconds) ->
  receive
  after Milliseconds -> ok
  end,
  ok.
