-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, start_timer/3]).

start() ->
  Timeout = 100,
  TimerReference = start_timer(Timeout, self(), message),
  %% Invalid option, actual option is `async`
  Options = [{sync, true}],
  test:caught(fun () ->
    cancel_timer(TimerReference, Options)
  end).
