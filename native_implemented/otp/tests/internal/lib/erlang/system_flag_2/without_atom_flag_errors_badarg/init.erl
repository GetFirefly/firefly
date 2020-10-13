-module(init).
-export([start/0]).
-import(erlang, [system_flag/2]).

start() ->
  test:each(fun
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(Flag) ->
  test:caught(fun () ->
    system_flag(Flag, [])
  end).
