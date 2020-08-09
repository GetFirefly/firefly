-module(init).
-export([start/0]).
-import(erlang, [binary_part/3, byte_size/1, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  ByteSize = byte_size(Binary),
  Start = ByteSize,
  Length = -ByteSize,
  BinaryPart = binary_part(Binary, Start, Length),
  display(BinaryPart).
