-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, display/1]).

start() ->
  TimerReference = test:reference(),
  Options = [{async, false}, {info, true}],
  display(cancel_timer(TimerReference, Options)).
