-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(process_flag(trap_exit, true)),
  display(process_flag(trap_exit, false)).
