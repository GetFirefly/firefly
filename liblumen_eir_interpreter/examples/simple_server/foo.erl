-module(foo).

-export([run/0]).

run() ->
    {ok, Pid} = 'Elixir.ExampleServer':start_link([]),
    ok = 'Elixir.GenServer':cast(Pid, {push, yay}),
    ok = 'Elixir.GenServer':cast(Pid, {push, yay2}),
    yay2 = 'Elixir.GenServer':call(Pid, pop, infinity),
    yay = 'Elixir.GenServer':call(Pid, pop, infinity).

