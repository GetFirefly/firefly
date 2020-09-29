-module(init).
-export([start/0]).
-import(erlang, [cancel_timer/2, display/1 , self/0, start_timer/3]).

start() ->
  Timeout = 100,
  TimerReference = start_timer(Timeout, self(), message),
  After = receive
            {timeout,TimerReference,Message} ->
              Message
           after
             105 ->
               after_instead_of_receive
           end,
  display(After),
  Options = [{async, false}, {info, true}],
  display(cancel_timer(TimerReference, Options)),
  display(cancel_timer(TimerReference, Options)).
