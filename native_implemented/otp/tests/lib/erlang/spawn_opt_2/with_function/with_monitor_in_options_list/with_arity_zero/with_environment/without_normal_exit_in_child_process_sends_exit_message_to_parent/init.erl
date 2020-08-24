-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  Environment = environment(),
  Options = [monitor],
  {_ChildPid, ChildMonitorReference} = spawn_opt(fun () ->
    exit(Environment)
  end, Options),
  receive
    {'DOWN', ChildMonitorReference, process, _, Reason} ->
      display({child, exited, Reason})
  after 10 ->
    display(timeout)
  end.

environment() ->
  abnormal.
