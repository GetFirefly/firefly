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
  Options = [{async, true}, {info, true}],
  display(cancel_timer(TimerReference, Options)),
  receive
    {cancel_timer, TimerReference, MillisecondsRemaining} ->
      display(is_integer(MillisecondsRemaining)),
      display(0 < MillisecondsRemaining);
    OtherMessage ->
      display({other_message, OtherMessage})
  after
    100 ->
      display(no_message_after_cancel)
  end,
  display(cancel_timer(TimerReference, Options)),
  receive
    {cancel_timer, TimerReference, Result} -> display(Result);
    OtherMessage -> display({other_message, OtherMessage})
  after
    100 ->
      display(no_message_after_cancel)
  end.
