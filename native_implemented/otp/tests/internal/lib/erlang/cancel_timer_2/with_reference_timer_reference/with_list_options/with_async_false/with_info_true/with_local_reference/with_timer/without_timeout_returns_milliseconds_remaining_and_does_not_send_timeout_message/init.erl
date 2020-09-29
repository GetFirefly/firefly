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
  Options = [{async, false}, {info, true}],
  MillisecondsRemaining = cancel_timer(TimerReference, Options),
  display(is_integer(MillisecondsRemaining)),
  display(0 < MillisecondsRemaining),
  display(MillisecondsRemaining =< 50),
  display(cancel_timer(TimerReference, Options)),
  After = receive
    Message -> Message
  after
    100 ->
      no_message_after_cancel
  end,
  display(After),
  display(cancel_timer(TimerReference, Options)).
