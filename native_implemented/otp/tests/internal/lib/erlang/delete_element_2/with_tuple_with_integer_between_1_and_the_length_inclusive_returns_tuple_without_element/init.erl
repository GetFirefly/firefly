-module(init).
-export([start/0]).
-import(erlang, [delete_element/2, display/1]).

start() ->
  display(delete_element(1, {1})),
  display(delete_element(1, {1, 2})),
  display(delete_element(2, {1, 2})).
