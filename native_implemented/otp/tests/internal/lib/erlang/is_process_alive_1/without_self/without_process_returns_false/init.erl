-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  FakePid = list_to_pid("<0.0.1>"),
  display(is_process_alive(FakePid)).
