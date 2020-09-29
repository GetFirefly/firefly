-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2]).

start() ->
  test:each(fun
    (Reference) when is_reference(Reference) -> ignore;
    (Term) -> test(Term)
  end).

test(TimerReference) ->
  Options = [],
  test:caught(fun () ->
    cancel_timer(TimerReference, Options)
  end).
