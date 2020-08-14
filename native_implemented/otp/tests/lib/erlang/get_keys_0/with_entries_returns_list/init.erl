-module(init).
-export([start/0]).
-import(erlang, [display/1, get_keys/1, put/2]).

start() ->
  Key = key,
  put(Key, value),
  display(get_keys()).

