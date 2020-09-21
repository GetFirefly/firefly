-module(init).
-export([start/0]).
-import(erlang, [display/1, erase/0, put/2]).

start() ->
  ProcessDictionary = erase(),
  display(ProcessDictionary).

