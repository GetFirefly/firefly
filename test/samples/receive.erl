-module(init).

-export([start/0]).

-import(erlang, [print/1]).

-spec start() -> ok | error.
start() ->
    Parent = self(),
    Loop = fun(Loop) ->
        receive
          {ping, Parent} ->
            print(<<"Sending pong..">>),
            Parent ! pong,
            Loop(Loop);

          {shutdown, Parent} ->
            print(<<"Shutting down..">>),
            Parent ! ok,
            ok
        end
    end,
    Pid = spawn(fun() -> Loop(Loop) end),
    Pid ! {ping, Parent},

    print(<<"Awaiting..">>),

    await(Parent, Pid).

await(Parent, Pid) ->
    receive
      pong ->
        print(<<"Received pong">>),
        Pid ! {shutdown, Parent},
        print(<<"Terminating child..">>),
        await(Parent, Pid);

      ok ->
        print(<<"Received ok">>),
        ok
    end.
