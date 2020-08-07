-module(init).
-export([start/0]).
-import(erlang, [atom_to_binary/2, display/1]).

start() ->
  display(atom_to_binary(one, latin1)),
  display(atom_to_binary(two, unicode)),
  display(atom_to_binary(three, utf8)).
