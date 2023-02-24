-module(init).
-export([start/0]).
-import(erlang, [binary_to_list/3, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  lists(Binary).

lists(Binary) ->
  lists(Binary, 1).

lists(Binary, Start) when Start == byte_size(Binary) ->
  lists(Binary, Start, Start);
lists(Binary, Start) ->
  lists(Binary, Start, Start),
  lists(Binary, Start + 1).

lists(Binary, Start, Stop) when Stop == byte_size(Binary) ->
  list(Binary, Start, Stop);
lists(Binary, Start, Stop) ->
  list(Binary, Start, Stop),
  lists(Binary, Start, Stop + 1).

list(Binary, Start, Stop) ->
  display(binary_to_list(Binary, Start, Stop)).
