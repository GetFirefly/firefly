-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, display/1]).

start() ->
  test:each(fun
    (Reference) when is_reference(Reference) -> ignore;
    (Term) -> test(Term)
  end).

test(TimerReference) ->
  test:caught(fun () ->
    Options = [{async, false}, {info, false}],
    cancel_timer(TimerReference, Options)
  end).
