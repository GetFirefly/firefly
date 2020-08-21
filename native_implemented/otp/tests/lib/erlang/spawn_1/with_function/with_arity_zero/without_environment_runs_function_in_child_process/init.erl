-module(init).
-export([start/0]).
-import(erlang, [display/1]).
-import(lumen, [is_big_integer/1, is_small_integer/1]).

start() ->
  ChildPid = spawn(fun () ->
    display(no_environment),
    receive
      Message -> Message
    after 10 ->
      ok
    end,
    ok
  end),
  MonitorRef = monitor(process, ChildPid),
  receive
    {'DOWN', MonitorRef, process, _, Info} ->
      display(Info)
    after 20 ->
      display(timeout)
  end,
  ok.
