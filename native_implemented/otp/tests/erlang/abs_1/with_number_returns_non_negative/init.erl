-module(init).
-compile({no_auto_import,[abs/1]}).
-export([start/0]).
-import(erlang, [abs/1, display/1]).

start() ->
  display(abs(-2.0)),
%%  display(abs(-1)),
  display(abs(0.0)),
%%  display(abs(0)),
%%  display(abs(1)).
  display(abs(2.0)).
