-module(init).
-export([start/0]).
-import(erlang, [atom_to_list/1, display/1]).

start() ->
  display(atom_to_list(one)),
  display(atom_to_list(two)),
  display(atom_to_list(three)).
