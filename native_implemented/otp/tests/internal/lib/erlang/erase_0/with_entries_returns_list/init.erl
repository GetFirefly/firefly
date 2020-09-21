-module(init).
-export([start/0]).
-import(erlang, [display/1, erase/0, put/2]).

start() ->
  put(key, value),
  ProcessDictionary = erase(),
  display(ProcessDictionary).

