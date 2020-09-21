-module(init).
-export([start/0]).
-import(erlang, [display/1, get/0, put/2]).

start() ->
  ProcessDictionary = get(),
  display(ProcessDictionary).

