-module(erlang).

-export([is_atom/1, is_binary/1, is_bitstring/1, is_boolean/1,
         is_float/1, is_function/1, is_function/2, is_integer/1,
         is_list/1, is_number/1, is_pid/1, is_port/1, is_map/1, is_record/2,
         nif_error/1, error/1, error/2, error/3,
         spawn_monitor/1, spawn_monitor/3, spawn_opt/4]).

-compile({no_auto_import,[spawn_link/1]}).
-compile({no_auto_import,[spawn_link/4]}).
-compile({no_auto_import,[spawn_opt/2]}).
-compile({no_auto_import,[spawn_opt/4]}).
-compile({no_auto_import,[spawn_opt/5]}).

%% error/1
%% Shadowed by erl_bif_types: erlang:error/1
-spec error(Reason) -> no_return() when
      Reason :: term().
error(_Reason) ->
    erlang:nif_error(undefined).

%% error/2
%% Shadowed by erl_bif_types: erlang:error/2
-spec error(Reason, Args) -> no_return() when
      Reason :: term(),
      Args :: [term()] | none.
error(_Reason, _Args) ->
    erlang:nif_error(undefined).

%% error/3
%% Shadowed by erl_bif_types: erlang:error/3
-spec error(Reason, Args, Options) -> no_return() when
      Reason :: term(),
      Args :: [term()] | none,
      Options :: [Option],
      Option :: {'error_info', ErrorInfoMap},
      ErrorInfoMap :: #{'cause' => term(),
                        'module' => module(),
                        'function' => atom()}.
error(_Reason, _Args, _Options) ->
    erlang:nif_error(undefined).


%% Shadowed by erl_bif_types: erlang:is_atom/1
-spec is_atom(Term) -> boolean() when
      Term :: term().
is_atom(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_binary/1
-spec is_binary(Term) -> boolean() when
      Term :: term().
is_binary(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_bitstring/1
-spec is_bitstring(Term) -> boolean() when
      Term :: term().
is_bitstring(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_boolean/1
-spec is_boolean(Term) -> boolean() when
      Term :: term().
is_boolean(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_float/1
-spec is_float(Term) -> boolean() when
      Term :: term().
is_float(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_function/1
-spec is_function(Term) -> boolean() when
      Term :: term().
is_function(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_function/2
-spec is_function(Term, Arity) -> boolean() when
      Term :: term(),
      Arity :: arity().
is_function(_Term, _Arity) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_integer/1
-spec is_integer(Term) -> boolean() when
      Term :: term().
is_integer(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_list/1
-spec is_list(Term) -> boolean() when
      Term :: term().
is_list(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_number/1
-spec is_number(Term) -> boolean() when
      Term :: term().
is_number(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_pid/1
-spec is_pid(Term) -> boolean() when
      Term :: term().
is_pid(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_map/1
-spec is_map(Term) -> boolean() when
      Term :: term().
is_map(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_port/1
-spec is_port(Term) -> boolean() when
      Term :: term().
is_port(_Term) ->
    erlang:nif_error(undefined).

%% Shadowed by erl_bif_types: erlang:is_record/2
-spec is_record(Term,RecordTag) -> boolean() when
      Term :: term(),
      RecordTag :: atom().
is_record(_Term,_RecordTag) ->
    erlang:nif_error(undefined).

%% nif_error/1
%% Shadowed by erl_bif_types: erlang:nif_error/1
-spec erlang:nif_error(Reason) -> no_return() when
      Reason :: term().
nif_error(_Reason) ->
    erlang:nif_error(undefined).

%% We must inline these functions so that the stacktrace points to
%% the correct function.
-compile({inline, [badarg_with_info/1]}).

-spec spawn_monitor(Fun) -> {pid(), reference()} when
      Fun :: function().
spawn_monitor(F) when erlang:is_function(F, 0) ->
    erlang:spawn_opt(erlang,apply,[F,[]],[monitor]);
spawn_monitor(F) ->
    badarg_with_info([F]).

-spec spawn_monitor(Module, Function, Args) -> {pid(), reference()} when
      Module :: module(),
      Function :: atom(),
      Args :: [term()].
spawn_monitor(M, F, A) when erlang:is_atom(M),
                            erlang:is_atom(F),
                            erlang:is_list(A) ->
    erlang:spawn_opt(M,F,A,[monitor]);
spawn_monitor(M, F, A) ->
    badarg_with_info([M,F,A]).


-type monitor_option() :: {'alias', 'explicit_unalias' | 'demonitor' | 'reply_demonitor'}
                        | {'tag', term()}.
-type max_heap_size() ::
        Size :: non_neg_integer()
        %% TODO change size => to := when -type maps support is finalized
      | #{ size => non_neg_integer(),
           kill => boolean(),
           error_logger => boolean() }.

-type priority_level() ::
      low | normal | high | max.

-type message_queue_data() ::
	off_heap | on_heap.

-type spawn_opt_option() ::
	link
      | monitor
      | {monitor, MonitorOpts :: [monitor_option()]}
      | {priority, Level :: priority_level()}
      | {fullsweep_after, Number :: non_neg_integer()}
      | {min_heap_size, Size :: non_neg_integer()}
      | {min_bin_vheap_size, VSize :: non_neg_integer()}
      | {max_heap_size, Size :: max_heap_size()}
      | {message_queue_data, MQD :: message_queue_data()}.

-spec spawn_opt(Module, Function, Args, Options) ->
          Pid | {Pid, MonitorRef} when
      Module :: module(),
      Function :: atom(),
      Args :: [term()],
      Options :: [spawn_opt_option()],
      Pid :: pid(),
      MonitorRef :: reference().
spawn_opt(_Module, _Function, _Args, _Options) ->
   erlang:nif_error(undefined).


%% garbage_collect/1
-spec garbage_collect(Pid) -> GCResult when
      Pid :: pid(),
      GCResult :: boolean().
garbage_collect(Pid) ->
    try
        erlang:garbage_collect(Pid, [])
    catch
        error:Error -> error_with_info(Error, [Pid])
    end.

-record(gcopt, {
    async = sync :: sync | {async, _},
    type = major % default major, can also be minor
    }).

%% garbage_collect/2
-spec garbage_collect(Pid, OptionList) -> GCResult | async when
      Pid :: pid(),
      RequestId :: term(),
      Option :: {async, RequestId} | {type, 'major' | 'minor'},
      OptionList :: [Option],
      GCResult :: boolean().
garbage_collect(Pid, OptionList)  ->
    try
        GcOpts = get_gc_opts(OptionList, #gcopt{}),
        case GcOpts#gcopt.async of
            {async, ReqId} ->
                erts_internal:request_system_task(
                        Pid, inherit, {garbage_collect, ReqId, GcOpts#gcopt.type}),
                async;
            sync ->
                case Pid == erlang:self() of
                    true ->
                        erts_internal:garbage_collect(GcOpts#gcopt.type);
                    false ->
                        ReqId = erlang:make_ref(),
                        erts_internal:request_system_task(
                                    Pid, inherit,
                                    {garbage_collect, ReqId, GcOpts#gcopt.type}),
                        receive
                            {garbage_collect, ReqId, GCResult} ->
                                GCResult
                        end
                end
        end
        catch
            throw:bad_option -> badarg_with_cause([Pid, OptionList], bad_option);
            error:_ -> badarg_with_info([Pid, OptionList])
    end.

%% gets async opt and verify valid option list
get_gc_opts([{async, _ReqId} = AsyncTuple | Options], GcOpt = #gcopt{}) ->
    get_gc_opts(Options, GcOpt#gcopt{ async = AsyncTuple });
get_gc_opts([{type, T} | Options], GcOpt = #gcopt{}) ->
    get_gc_opts(Options, GcOpt#gcopt{ type = T });
get_gc_opts([], GcOpt) ->
    GcOpt;
get_gc_opts(_, _) ->
    erlang:throw(bad_option).


badarg_with_info(Args) ->
    erlang:error(badarg, Args, [{error_info, #{module => erl_erts_errors}}]).

badarg_with_cause(Args, Cause) ->
    erlang:error(badarg, Args, [{error_info, #{module => erl_erts_errors,
                                              cause => Cause}}]).

