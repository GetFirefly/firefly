-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Binary = <<>>,
  Length = 0,
  Start = 1,
  display(Start > byte_size(Binary)),
  StartLength = {Start, Length},
  test:caught(fun () ->
    binary_part(Binary, StartLength)
  end).
