-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  LinkCountBefore = link_count(self()),
  display(link(self())),
  LinkCountAfter = link_count(self()),
  display(LinkCountBefore == LinkCountAfter).

link_count(Pid) ->
   {links, Links} = process_info(Pid, links),
   length(Links).
