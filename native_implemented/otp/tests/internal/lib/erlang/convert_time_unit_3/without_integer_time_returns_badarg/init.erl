-module(init).
-export([start/0]).
-import(erlang, [convert_time_unit/3]).

start() ->
  test:each(fun
    (Integer) when is_integer(Integer) -> ignore;
    (Term) -> test(Term)
  end).

test(Time) ->
  test:caught(fun () ->
    convert_time_unit(Time, native, native)
  end).
