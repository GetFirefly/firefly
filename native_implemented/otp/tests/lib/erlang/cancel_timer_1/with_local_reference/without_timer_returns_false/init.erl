-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/1, display/1, make_ref/0]).

start() ->
  TimerReference = make_ref(),
  display(cancel_timer(TimerReference)).
