-module(init).

-export([start/0]).

-import(erlang, [print/1]).

-spec start() -> ok | error.
start() ->
  Binary = <<"Hello, world!">>,
  print(Binary).
