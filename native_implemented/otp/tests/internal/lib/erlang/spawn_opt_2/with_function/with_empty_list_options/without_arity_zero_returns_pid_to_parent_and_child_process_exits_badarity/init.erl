-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [log_exit/1]).

start() ->
  log_exit(false),
  Options = [],
  ChildPid = spawn_opt(fun (A) ->
    display(A)
  end, Options),
  MonitorRef = monitor(process, ChildPid),
  receive
    {'DOWN', MonitorRef, process, _, Info} ->
      case Info of
        %% FIXME https://github.com/lumen/lumen/issues/548
        {badarity = Reason, FunArgs} -> display(Reason);
        Other -> display(Other)
      end
    after 20 ->
      display(timeout)
  end,
  ok.
