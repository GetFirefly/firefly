-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, float_to_list/1, float_to_list/2, make_tuple/2]).

start() ->
  Options = options(),
  Zero = 0.0,
  display(float_to_list(Zero)),
  display(float_to_list(Zero, Options)),
  OneTenth = 0.1,
  display(float_to_list(OneTenth)),
  display(float_to_list(OneTenth, Options)).

options() ->
  Digits = 20,
  Scientific = pair(scientific, Digits),
  [Scientific].

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).
