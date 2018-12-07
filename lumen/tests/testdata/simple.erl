-module(simple).

-export([start/0, start/1]).

%% Spawns a process which prints a couple messages then exits
start() ->
    start([]).

%% Spawns a process with the given args, which prints a couple messages then exits
start(Args) when is_list(Args) ->
    Parent = self(),
    Child = spawn_link(fun() ->
      Parent ! {self(), {ok, []}},
      receive
          {Parent, exit} ->
              ok;
          {Parent, Msg} ->
              io:format("got ~p~n", [Msg])
      end
    end),
    receive
        {Child, {ok, _}} ->
            Child ! started,
            Child ! exit;
        {Child, Other} ->
            exit(Other)
    end.
