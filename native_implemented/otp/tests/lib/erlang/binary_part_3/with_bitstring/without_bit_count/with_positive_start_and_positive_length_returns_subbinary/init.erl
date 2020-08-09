-module(init).
-export([start/0]).
-import(erlang, [binary_part/3, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  Start = 1,
  Length = 1,
  BinaryPart = binary_part(Binary, Start, Length),
  display(BinaryPart).
