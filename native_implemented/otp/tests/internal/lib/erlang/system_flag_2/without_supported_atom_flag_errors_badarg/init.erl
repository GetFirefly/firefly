-module(init).
-export([start/0]).
-import(erlang, [system_flag/2]).

start() ->
  test:caught(fun () ->
    system_flag(unsupported_flag, [])
  end).
