-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, display/1, is_integer/1, self/0, start_timer/3]).

start() ->
  Timeout = 100,
  TimerReference = start_timer(Timeout, self(), message),
  Midway = receive
             Message -> Message
           after
             50 ->
               no_message_at_midway
           end,
  display(Midway),
  Options = [{async, true}],
  display(cancel_timer(TimerReference, Options)),
  receive
    {cancel_timer, TimerReference, MillisecondsRemaining} ->
      display(is_integer(MillisecondsRemaining)),
      display(0 < MillisecondsRemaining);
     OtherMessage -> display({other_message, OtherMessage})
  after 10 ->
    display(timeout)
  end,
  display(cancel_timer(TimerReference, Options)),
  receive
    {cancel_timer, TimerReference, Result} -> display(Result);
    SecondOtherMessage -> display({other_message, SecondOtherMessage})
  after 10 ->
    display(timeout)
  end.

