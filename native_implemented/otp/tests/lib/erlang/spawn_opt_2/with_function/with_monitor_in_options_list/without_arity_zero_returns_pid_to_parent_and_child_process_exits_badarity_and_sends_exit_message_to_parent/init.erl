-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  Options = [monitor],
  {_ChildPid, ChildMonitorReference} = spawn_opt(fun (A) ->
    display(A)
  end, Options),
  receive
    %% FIXME https://github.com/lumen/lumen/issues/548
    {'DOWN', ChildMonitorReference, process, _, {Reason, Details}} ->
      display({child, exited, Reason})
    after 10 ->
      display(timeout)
  end.
