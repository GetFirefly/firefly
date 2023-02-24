%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: true
-module(init).

-export([boot/1]).

boot(_) ->
    Child = spawn(fun () -> loop() end),
    Child ! {self(), ping},
    receive
        {Child, pong} ->
            erlang:display(true)
    after
        5000 ->
            erlang:display(timeout)
    end.

loop() ->
    receive
        {Pid, ping} ->
            Pid ! {self(), pong},
            ok;
        Other ->
            erlang:display(Other),
            loop()
    end.
