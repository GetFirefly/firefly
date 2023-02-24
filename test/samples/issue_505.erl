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
  display(has_no_messages()),
  Options = [info],
  display(demonitor(MonitorReference, Options)),
  ChildPid ! next,
  receive
     child_done -> ok
  end,
  display(has_no_messages()).

has_no_messages() ->
  HasNoMessages = receive
    _ -> false
  after
    0 -> true
  end,
  HasNoMessages.
