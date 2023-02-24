-module(init).
-export([start/0, counter/2]).
-import(erlang, [display/1]).

start() ->
  [_Program | Args] = init:get_plain_arguments(),
  case Args of
    [CountBinary | Options] ->
      Count = binary_to_integer(CountBinary),
      Output = case Options of
        [<<"--quiet">>] -> fun (_Tuple) ->
          ok
        end;
        [] -> fun (Tuple) ->
          display(Tuple)
        end
      end,
      spawn_chain(Count, Output)
  end.

spawn_chain(Count, Output) when Count > 0 ->
  display({self(), count, to, Count}),
  create_processes(Count, Output).

create_processes(Count, Output) ->
  Last = reduce(1, Count, self(), Output),
  Last ! 0,
  receive
    FinalAnswer when is_integer(FinalAnswer) ->
      display({result, is, FinalAnswer})
  end.

reduce(Acc, Final, Receiver, Output) ->
  Sender = spawn(init, counter, [Receiver, Output]),
  case Acc of
    Final -> Sender;
    _ -> reduce(Acc + 1, Final, Sender, Output)
  end.

counter(Receiver, Output) ->
  receive
    N ->
      apply(Output, [{self(), received, N}]),
      Sent = Receiver ! N + 1,
      apply(Output, [{self(), sent, Sent, to, Receiver}])
  end.
