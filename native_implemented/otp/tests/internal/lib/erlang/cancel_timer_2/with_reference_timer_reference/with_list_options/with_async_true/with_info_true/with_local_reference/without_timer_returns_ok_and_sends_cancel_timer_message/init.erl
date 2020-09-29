-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, display/1]).

start() ->
  TimerReference = test:reference(),
  Options = [{async, true}, {info, true}],
  display(cancel_timer(TimerReference, Options)),
  receive
    {cancel_timer, TimerReference, Result} -> display(Result);
    OtherMessage -> display({other_message, OtherMessage})
  after 10 ->
    display(timeout)
  end.
