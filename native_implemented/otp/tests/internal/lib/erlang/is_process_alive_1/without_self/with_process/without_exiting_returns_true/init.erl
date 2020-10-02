-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  ChildPid = spawn(fun () ->
    wait_to_shutdown()
  end),
  display(is_process_alive(ChildPid)),
  shutdown(ChildPid).

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
