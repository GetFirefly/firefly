-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Bitstring) when is_bitstring(Bitstring) -> ignore;
    (Term) -> test(Term)
  end).

test(Binary) ->
  StartLength = {0, 0},
  test:caught(fun () ->
    binary_part(Binary, StartLength)
  end).
