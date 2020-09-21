-module(init).
-export([start/0]).
-import(erlang, [append_element/2, display/1, get_keys/1, make_tuple/2, put/2]).

start() ->
  put(mary, pair(1, 2)),
  put(had, pair(1, 2)),
  put(a, pair(1, 2)),
  put(little, pair(1, 2)),
  put(dog, pair(1, 3)),
  put(lamb, pair(1, 2)),
  Keys = get_keys(pair(1, 2)),
  display(lists:member(mary, Keys)),
  display(lists:member(had, Keys)),
  display(lists:member(a, Keys)),
  display(lists:member(little, Keys)),
  display(lists:member(lamb, Keys)),
  display(not lists:member(dog, Keys)).

%% FIXME work around tuple lowering bug
pair(Key, Value) ->
  Empty = make_tuple(0, []),
  KeyTuple = append_element(Empty, Key),
  append_element(KeyTuple, Value).

