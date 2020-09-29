-module(init).
-export([start/0]).
-import(erlang, [convert_time_unit/3]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Atom) when is_atom(Atom) -> ignore;
    (Term) -> test(Term)
  end).

test(FromUnit) ->
  test:caught(fun () ->
    convert_time_unit(0, FromUnit, native)
  end).
