-module(init).
-export([start/0]).
-import(erlang, [demonitor/1, display/1, make_ref/0]).

start() ->
  Reference = make_ref(),
  display(demonitor(Reference)).
