-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Options = [monitor],
  {_ChildPid, ChildMonitorReference} = spawn_opt(fun () ->
    ok
  end, Options),
  receive
    {'DOWN', ChildMonitorReference, process, _, Reason} ->
      display({child, exited, Reason})
  after 10 ->
    display(timeout)
  end.
