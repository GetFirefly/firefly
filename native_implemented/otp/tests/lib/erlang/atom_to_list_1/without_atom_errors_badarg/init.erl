-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(Atom) ->
  test:caught(fun () ->
    atom_to_list(Atom)
  end).
