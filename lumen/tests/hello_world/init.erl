-module(init).

-export([start/0]).

-import(erlang, [display/1]).

-spec start() -> ok | error.
start() ->
  Binary = <<"Hello, world!">>,
  display(Binary).
