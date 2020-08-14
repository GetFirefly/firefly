-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, get_keys/0, make_tuple/2, put/2]).

start() ->
  put(dog, pair(animal, 1)),
  put(cow, pair(animal, 2)),
  put(lamb, pair(animal, 3)),
  Keys = get_keys(),
  display(lists:member(dog, Keys)),
  display(lists:member(cow, Keys)),
  display(lists:member(lamb, Keys)).

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).
