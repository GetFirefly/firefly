-module(init).
-export([start/0]).
-import(erlang, [display/1, module_loaded/1]).

start() ->
  display(module_loaded(non_existing)).
