-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:caught(fun () ->
    display(binary:encode_unsigned(-1))
  end),
  test:caught(fun () ->
    display(binary:encode_unsigned(-70368744177664))
  end).