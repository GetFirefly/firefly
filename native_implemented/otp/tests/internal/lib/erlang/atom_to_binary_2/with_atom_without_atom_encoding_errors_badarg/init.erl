-module(init).
-export([start/0]).
-import(erlang, [atom_to_binary/2, display/1]).

start() ->
  test:each(fun
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(Encoding) ->
  Atom = test:atom(),
  test:caught(fun () ->
    atom_to_binary(Atom, Encoding)
  end).
