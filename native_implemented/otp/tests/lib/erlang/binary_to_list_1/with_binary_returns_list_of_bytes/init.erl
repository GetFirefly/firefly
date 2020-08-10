-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  display(binary_to_list(<<>>)),
  display(binary_to_list(<<0, 1>>)),
  display(binary_to_list(<<"ascii">>)).
