-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  test:caught(fun () ->
    display(binary:encode_unsigned(nil))
  end),
  test:caught(fun () ->
    display(binary:encode_unsigned(foo))
  end),
  test:caught(fun () ->
    display(binary:encode_unsigned("foo"))
  end).