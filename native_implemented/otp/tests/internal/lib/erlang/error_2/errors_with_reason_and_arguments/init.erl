-module(init).
-export([start/0]).

start() ->
  test:caught(fun () ->
    error(reason, [1, 2])
  end).
