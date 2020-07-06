-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(true andalso right()).

right() ->
  right.
