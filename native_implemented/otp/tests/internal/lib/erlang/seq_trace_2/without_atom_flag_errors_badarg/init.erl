-module(init).
-export([start/0]).
-import(erlang, [seq_trace/2]).

start() ->
  test:each(fun
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(Flag) ->
  test:caught(fun () ->
    seq_trace(Flag, false)
  end).
