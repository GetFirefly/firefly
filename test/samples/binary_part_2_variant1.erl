-module(init).
-export([start/0]).
-import(erlang, [binary_part/2, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  Start = 1,
  Length = 1,
  StartLength = {Start, Length},
  BinaryPart = binary_part(Binary, StartLength),
  display(BinaryPart).
