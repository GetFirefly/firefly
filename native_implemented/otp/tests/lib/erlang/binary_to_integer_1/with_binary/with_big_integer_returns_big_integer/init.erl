-module(init).
-export([start/0]).
-import(erlang, [binary_to_integer/1, display/1]).

start() ->
  display(binary_to_integer(<<"-9223372036854775808">>)),
  display(binary_to_integer(<<"9223372036854775807">>)).
