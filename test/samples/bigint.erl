-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  % i64::min_value()
  display(-9223372036854775808),
  % i64::max_value()
  display(9223372036854775807).
