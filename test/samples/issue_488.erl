-module(init).
-export([start/0]).

start() ->
  receive
    Message -> Message
  end,
  ok.
