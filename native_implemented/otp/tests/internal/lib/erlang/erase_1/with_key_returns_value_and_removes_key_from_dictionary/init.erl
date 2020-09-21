-module(init).
-export([start/0]).
-import(erlang, [display/1, erase/1, put/2]).

start() ->
  Key = key,
  put(Key, value),
  display(erase(Key)),
  display(get(Key)).

