-module(init).
-export([start/0]).
-import(erlang, [seq_trace_info/1]).

start() ->
  test:each(fun
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(Item) ->
  test:caught(fun () ->
    seq_trace_info(Item)
  end).
