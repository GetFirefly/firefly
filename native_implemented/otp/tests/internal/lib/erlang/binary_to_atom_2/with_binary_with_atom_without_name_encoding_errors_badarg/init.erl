-module(init).
-export([start/0]).

start() ->
  test:each(fun
              (Atom) when is_atom(Atom) -> ignore;
              (Term) -> test(Term)
            end).

test(Encoding) ->
  Binary = <<>>,
  test:caught(fun () ->
    binary_to_atom(Binary, Encoding)
  end).
