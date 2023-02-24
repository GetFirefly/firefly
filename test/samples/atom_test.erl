-module(init).
-export([start/0]).

start() ->
  Atom = atom,
  true = is_atom(Atom),
  Atom.

