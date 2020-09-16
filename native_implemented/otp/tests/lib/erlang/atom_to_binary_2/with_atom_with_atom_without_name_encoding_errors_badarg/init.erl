-module(init).
-export([start/0]).
-import(erlang, [atom_to_binary/2, display/1]).

start() ->
  Atom = test:atom(),
  Encoding = not_an_encoding,
  true = is_atom(Encoding),
  test:caught(fun () ->
    atom_to_binary(Atom, Encoding)
  end).
