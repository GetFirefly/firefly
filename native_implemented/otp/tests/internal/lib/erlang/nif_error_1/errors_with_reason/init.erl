-module(init).
-export([start/0]).
-import(erlang, [nif_error/1]).

start() ->
  test:caught(fun () ->
    nif_error(reason)
  end).
