-module(init).
-export([start/0]).
-import(erlang, [binary_part/2, byte_size/1, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  Start = 0,
  Length = byte_size(Binary),
  BinaryPart = binary_part(Binary, Start, Length),
  display(BinaryPart).
