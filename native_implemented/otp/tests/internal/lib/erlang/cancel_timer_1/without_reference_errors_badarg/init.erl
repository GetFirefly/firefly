-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/1]).

start() ->
  test:each(fun
    (Reference) when is_reference(Reference) -> ignore;
    (Term) -> test(term)
  end).

test(TimerReference) ->
  test:caught(fun () ->
    cancel_timer(TimerReference)
  end).
