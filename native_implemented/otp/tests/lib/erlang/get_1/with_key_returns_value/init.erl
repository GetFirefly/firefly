-module(init).
-export([start/0]).
-import(erlang, [display/1, get/1, put/2]).

start() ->
  Key = key,
  put(Key, value),
  display(get(Key)).

