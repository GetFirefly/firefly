-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  Dividend = test_float:min(),
  Divisor = 0.1,
  display(Dividend / Divisor).
