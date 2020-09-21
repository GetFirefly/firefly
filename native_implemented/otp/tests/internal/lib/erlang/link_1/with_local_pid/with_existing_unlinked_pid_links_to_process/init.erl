-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  ChildPid = spawn(fun () ->
    wait_to_shutdown()
  end),
  LinkCountBefore = link_count(self()),
  display(link(ChildPid)),
  LinkCountAfter = link_count(self()),
  display(LinkCountBefore + 1 == LinkCountAfter),
  unlink(ChildPid),
  shutdown(ChildPid).

link_count(Pid) ->
   {links, Links} = process_info(Pid, links),
   length(Links).

shutdown(Pid) ->
  Pid ! shutdown.

wait_to_shutdown() ->
  receive
    shutdown -> ok
  end.
