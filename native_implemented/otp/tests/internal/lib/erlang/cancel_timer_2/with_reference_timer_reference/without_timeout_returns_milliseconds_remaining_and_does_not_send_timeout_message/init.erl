-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2]).

start() ->
  test:each(fun
    (List) when is_list(List) -> ignore;
    (Term) -> test(Term)
  end).

test(Options) ->
  test:caught(fun () ->
    TimerReference = test:reference(),
    cancel_timer(TimerReference, Options)
  end).
