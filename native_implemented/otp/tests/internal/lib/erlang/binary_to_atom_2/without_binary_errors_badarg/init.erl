-module(init).
-export([start/0]).

start() ->
  test:each(fun
    (Binary) when is_binary(Binary) -> ignore;
    (Term) -> test(Term)
  end).

test(Binary) ->
  Encoding = unicode,
  test:caught(fun () ->
    binary_to_atom(Binary, Encoding)
  end).
