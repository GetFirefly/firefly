-module(erts_internal).

-export([is_process_alive/1, is_process_alive/2]).

-spec erts_internal:is_process_alive(Pid) -> boolean() when
      Pid :: pid().
is_process_alive(Pid) ->
    Ref = make_ref(),
    erts_internal:is_process_alive(Pid, Ref),
    receive
        {Ref, Res} ->
            Res
    end.

-spec erts_internal:is_process_alive(Pid, Ref) -> 'ok' when
      Pid :: pid(),
      Ref :: reference().
is_process_alive(_Pid, _Ref) ->
    erlang:nif_error(undefined).
