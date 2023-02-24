-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  ChildPid = spawn(fun () ->
    display("In child"),
    receive
      Message -> Message
    after 10 ->
      ok
    end,
    ok
  end),
  MonitorRef = monitor(process, ChildPid),
  receive
    {'DOWN', MonitorRef, process, ChildPid, Info} ->
      display(Info)
  end,
  ok.
