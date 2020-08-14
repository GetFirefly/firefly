-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, get_keys/1, make_tuple/2, put/2]).

start() ->
  FirstKeyWithValue = first_key_with_value,
  KeyWithOtherValue = key_with_other_value,
  SecondKeyWithValue = second_key_with_value,
  Value = value,
  put(FirstKeyWithValue, Value),
  put(KeyWithOtherValue, other_value),
  put(SecondKeyWithValue, Value),
  Keys = get_keys(Value),
  display(lists:member(FirstKeyWithValue, Keys)),
  display(lists:member(SecondKeyWithValue, Keys)),
  display(lists:member(KeyWithOtherValue, Keys)).

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).

