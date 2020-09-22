-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Binary = <<>>,
  Length = 0,
  Start = byte_size(Binary) + 1,
  display(Start > byte_size(Binary)),
  test:caught(fun () ->
    binary_part(Binary, Start, Length)
  end).
