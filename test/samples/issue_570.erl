-module(init).
-export([start/0, spawn_chain/1, counter/1, create_processes/1]).
-import(erlang, [display/1]).

start() ->
  [_Program | Args] = init:get_plain_arguments(),
  case Args of
    [CountBinary] ->
      Count = binary_to_integer(CountBinary),
      spawn_chain(Count)
  end.

spawn_chain(Count) when Count > 0 ->
  display({self(), count, to, Count}),
  create_processes(Count).

create_processes(Count) ->
  Last = reduce(1, Count, self()),
  Last ! 0,
  receive
    FinalAnswer when is_integer(FinalAnswer) ->
      display({result, is, FinalAnswer})
  end.

reduce(Final, Final, Receiver) ->
  spawn(?MODULE, counter, [Receiver]);
reduce(Acc, Final, Receiver) ->
  Sender = spawn(?MODULE, counter, [Receiver]),
  reduce(Acc + 1, Final, Sender).

counter(Receiver) ->
  receive
    N ->
      display({self(), received, N}),
      Sent = Receiver ! N + 1,
      display({self(), sent, Sent, to, Receiver})
  end.
