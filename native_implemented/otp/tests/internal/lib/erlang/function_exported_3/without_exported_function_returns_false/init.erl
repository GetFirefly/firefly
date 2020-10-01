-module(init).
-export([start/0]).
-import(erlang, [display/1, function_exported/3]).

start() ->
  Module = init,
  Function = start,
  Arity = 1,
  display(function_exported(Module, Function, Arity)).
