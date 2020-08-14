-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, float_to_list/2, make_tuple/2]).

start() ->
  Options = options(),
  display(float_to_list(1.0, Options)).

options() ->
  Digits = 2,
  Decimals = pair(decimals, Digits),
  [Decimals, compact].

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).
