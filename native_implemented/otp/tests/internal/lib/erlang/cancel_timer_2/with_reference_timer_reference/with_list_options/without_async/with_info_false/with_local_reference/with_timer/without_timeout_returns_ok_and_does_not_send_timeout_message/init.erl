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
  Options = [{info, false}],
  display(cancel_timer(TimerReference, Options)),
  receive
    MessageAfterCancel -> display({message_after_first_cancel, MessageAfterCancel})
  after 10 ->
    display(no_message_after_first_cancel)
  end,
  display(cancel_timer(TimerReference, Options)),
  receive
    MessageAfterCancel -> display({message_after_second_cancel, MessageAfterCancel})
  after 10 ->
    display(no_message_after_second_cancel)
  end.
