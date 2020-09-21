-module(init).
-export([start/0]).
-import(erlang, [binary_to_list/3, display/1]).

start() ->
  Binary = <<0, 1, 2>>,
  lists(Binary).

lists(Binary) ->
  lists(Binary, 1).

lists(Binary, Start) ->
  case byte_size(Binary) of
    Start ->
      lists(Binary, Start, Start);
    _ ->
      lists(Binary, Start, Start),
      lists(Binary, Start + 1)
  end.

lists(Binary, Start, Stop) ->
  case byte_size(Binary) of
    Stop ->
      list(Binary, Start, Stop);
    _ ->
      list(Binary, Start, Stop),
      lists(Binary, Start, Stop + 1)
  end.

list(Binary, Start, Stop) ->
  display(binary_to_list(Binary, Start, Stop)).
