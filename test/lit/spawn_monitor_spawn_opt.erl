-module(init).

-export([boot/1]).

boot(_) ->
    {ParentPid, ParentMonitorReference} = spawn_monitor(fun () ->
        Options = [link, monitor],
        spawn_opt(fun (_) ->
            ok
        end, Options),
        receive
            shutdown ->
                ok
        end
    end),
    receive
        {'DOWN', ParentMonitorReference, process, _, Info} ->
        case Info of
            {Reason = badarity, _FunArgs} ->
                erlang:display({parent, Reason});
            _ ->
                erlang:display({parent, Info})
        end
    after
        10 ->
            erlang:display({parent, alive, is_process_alive(ParentPid)})
    end.
