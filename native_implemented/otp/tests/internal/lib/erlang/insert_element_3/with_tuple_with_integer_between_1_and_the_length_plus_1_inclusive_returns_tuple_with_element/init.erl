-module(init).
-export([start/0]).
-import(erlang, [display/1, insert_element/3]).

start() ->
  display(insert_element(1, {}, inserted_element)),
  display(insert_element(1, {1}, inserted_element)),
  display(insert_element(2, {1}, inserted_element)),
  display(insert_element(1, {1, 2}, inserted_element)),
  display(insert_element(2, {1, 2}, inserted_element)),
  display(insert_element(3, {1, 2}, inserted_element)).
