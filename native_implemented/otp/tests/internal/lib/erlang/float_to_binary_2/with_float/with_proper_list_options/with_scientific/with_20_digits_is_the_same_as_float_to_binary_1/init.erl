-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, float_to_binary/1, float_to_binary/2, make_tuple/2]).

start() ->
  Options = options(),
  Zero = 0.0,
  display(float_to_binary(Zero) == float_to_binary(Zero, Options)),
  OneTenth = 0.1,
  display(float_to_binary(OneTenth) == float_to_binary(OneTenth, Options)).

options() ->
  Digits = 20,
  Scientific = pair(scientific, Digits),
  [Scientific].

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).
