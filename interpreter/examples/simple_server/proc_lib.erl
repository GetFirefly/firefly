-file("proc_lib.erl", 1).

-module(proc_lib).

-export([spawn/1,
         spawn_link/1,
         spawn/2,
         spawn_link/2,
         spawn/3,
         spawn_link/3,
         spawn/4,
         spawn_link/4,
         spawn_opt/2,
         spawn_opt/3,
         spawn_opt/4,
         spawn_opt/5,
         start/3,
         start/4,
         start/5,
         start_link/3,
         start_link/4,
         start_link/5,
         hibernate/3,
         init_ack/1,
         init_ack/2,
         init_p/3,
         init_p/5,
         format/1,
         format/2,
         format/3,
         report_cb/2,
         initial_call/1,
         translate_initial_call/1,
         stop/1,
         stop/3]).

-export([wake_up/3]).

-export_type([spawn_option/0]).

-file("/home/hansihe/.asdf/plugins/erlang/kerl-home/builds/asdf_21.2.4/"
      "otp_src_21.2.4/lib/stdlib/src/../../kernel/include/logger.hrl",
      1).

-file("proc_lib.erl", 44).

-type priority_level() :: high | low | max | normal.

-type max_heap_size() ::
          non_neg_integer() |
          #{size => non_neg_integer(),
            kill => true,
            error_logger => true}.

-type spawn_option() ::
          link |
          monitor |
          {priority, priority_level()} |
          {max_heap_size, max_heap_size()} |
          {min_heap_size, non_neg_integer()} |
          {min_bin_vheap_size, non_neg_integer()} |
          {fullsweep_after, non_neg_integer()} |
          {message_queue_data, off_heap | on_heap | mixed}.

-type dict_or_pid() ::
          pid() |
          (ProcInfo :: [_]) |
          {X :: integer(), Y :: integer(), Z :: integer()}.

-spec spawn(Fun) -> pid() when Fun :: function().

spawn(F) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn(proc_lib, init_p, [Parent,Ancestors,F]).

-spec spawn(Module, Function, Args) -> pid()
               when
                   Module :: module(),
                   Function :: atom(),
                   Args :: [term()].

spawn(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn(proc_lib, init_p, [Parent,Ancestors,M,F,A]).

-spec spawn_link(Fun) -> pid() when Fun :: function().

spawn_link(F) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    spawn_link(proc_lib, init_p, [Parent,Ancestors,F]).

-spec spawn_link(Module, Function, Args) -> pid()
                    when
                        Module :: module(),
                        Function :: atom(),
                        Args :: [term()].

spawn_link(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn_link(proc_lib, init_p, [Parent,Ancestors,M,F,A]).

-spec spawn(Node, Fun) -> pid() when Node :: node(), Fun :: function().

spawn(Node, F) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn(Node, proc_lib, init_p, [Parent,Ancestors,F]).

-spec spawn(Node, Module, Function, Args) -> pid()
               when
                   Node :: node(),
                   Module :: module(),
                   Function :: atom(),
                   Args :: [term()].

spawn(Node, M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn(Node, proc_lib, init_p, [Parent,Ancestors,M,F,A]).

-spec spawn_link(Node, Fun) -> pid()
                    when Node :: node(), Fun :: function().

spawn_link(Node, F) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn_link(Node, proc_lib, init_p, [Parent,Ancestors,F]).

-spec spawn_link(Node, Module, Function, Args) -> pid()
                    when
                        Node :: node(),
                        Module :: module(),
                        Function :: atom(),
                        Args :: [term()].

spawn_link(Node, M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    erlang:spawn_link(Node, proc_lib, init_p, [Parent,Ancestors,M,F,A]).

-spec spawn_opt(Fun, SpawnOpts) -> pid()
                   when Fun :: function(), SpawnOpts :: [spawn_option()].

spawn_opt(F, Opts) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    check_for_monitor(Opts),
    erlang:spawn_opt(proc_lib, init_p, [Parent,Ancestors,F], Opts).

-spec spawn_opt(Node, Function, SpawnOpts) -> pid()
                   when
                       Node :: node(),
                       Function :: function(),
                       SpawnOpts :: [spawn_option()].

spawn_opt(Node, F, Opts) when is_function(F) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    check_for_monitor(Opts),
    erlang:spawn_opt(Node, proc_lib, init_p, [Parent,Ancestors,F], Opts).

-spec spawn_opt(Module, Function, Args, SpawnOpts) -> pid()
                   when
                       Module :: module(),
                       Function :: atom(),
                       Args :: [term()],
                       SpawnOpts :: [spawn_option()].

spawn_opt(M, F, A, Opts) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    check_for_monitor(Opts),
    erlang:spawn_opt(proc_lib, init_p, [Parent,Ancestors,M,F,A], Opts).

-spec spawn_opt(Node, Module, Function, Args, SpawnOpts) -> pid()
                   when
                       Node :: node(),
                       Module :: module(),
                       Function :: atom(),
                       Args :: [term()],
                       SpawnOpts :: [spawn_option()].

spawn_opt(Node, M, F, A, Opts) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    check_for_monitor(Opts),
    erlang:spawn_opt(Node, proc_lib, init_p, [Parent,Ancestors,M,F,A], Opts).

check_for_monitor(SpawnOpts) ->
    case lists:member(monitor, SpawnOpts) of
        true ->
            error(badarg);
        false ->
            false
    end.

spawn_mon(M, F, A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    spawn_monitor(proc_lib, init_p, [Parent,Ancestors,M,F,A]).

spawn_opt_mon(M, F, A, Opts) when is_atom(M), is_atom(F), is_list(A) ->
    Parent = get_my_name(),
    Ancestors = get_ancestors(),
    check_for_monitor(Opts),
    spawn_opt(proc_lib,
              init_p,
              [Parent,Ancestors,M,F,A],
              [monitor|Opts]).

-spec hibernate(Module, Function, Args) -> no_return()
                   when
                       Module :: module(),
                       Function :: atom(),
                       Args :: [term()].

hibernate(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    erlang:hibernate(proc_lib, wake_up, [M,F,A]).

ensure_link(SpawnOpts) ->
    case lists:member(link, SpawnOpts) of
        true ->
            SpawnOpts;
        false ->
            [link|SpawnOpts]
    end.

-spec init_p(pid(), [pid()], function()) -> term().

init_p(Parent, Ancestors, Fun) when is_function(Fun) ->
    put('$ancestors', [Parent|Ancestors]),
    Mfa = erlang:fun_info_mfa(Fun),
    put('$initial_call', Mfa),
    try
        Fun()
    catch
        Class:Reason:Stacktrace ->
            exit_p(Class, Reason, Stacktrace)
    end.

-spec init_p(pid(), [pid()], atom(), atom(), [term()]) -> term().

init_p(Parent, Ancestors, M, F, A)
    when is_atom(M), is_atom(F), is_list(A) ->
    put('$ancestors', [Parent|Ancestors]),
    put('$initial_call', trans_init(M, F, A)),
    init_p_do_apply(M, F, A).

init_p_do_apply(M, F, A) ->
    try
        apply(M, F, A)
    catch
        Class:Reason:Stacktrace ->
            exit_p(Class, Reason, Stacktrace)
    end.

-spec wake_up(atom(), atom(), [term()]) -> term().

wake_up(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    try
        apply(M, F, A)
    catch
        Class:Reason:Stacktrace ->
            exit_p(Class, Reason, Stacktrace)
    end.

exit_p(Class, Reason, Stacktrace) ->
    case get('$initial_call') of
        {M,F,A} when is_atom(M), is_atom(F), is_integer(A) ->
            MFA = {M,F,make_dummy_args(A, [])},
            crash_report(Class, Reason, MFA, Stacktrace),
            erlang:raise(exit,
                         exit_reason(Class, Reason, Stacktrace),
                         Stacktrace);
        _ ->
            crash_report(Class, Reason, [], Stacktrace),
            erlang:raise(exit,
                         exit_reason(Class, Reason, Stacktrace),
                         Stacktrace)
    end.

exit_reason(error, Reason, Stacktrace) ->
    {Reason,Stacktrace};
exit_reason(exit, Reason, _Stacktrace) ->
    Reason;
exit_reason(throw, Reason, Stacktrace) ->
    {{nocatch,Reason},Stacktrace}.

-spec start(Module, Function, Args) -> Ret
               when
                   Module :: module(),
                   Function :: atom(),
                   Args :: [term()],
                   Ret :: term() | {error, Reason :: term()}.

start(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    start(M, F, A, infinity).

-spec start(Module, Function, Args, Time) -> Ret
               when
                   Module :: module(),
                   Function :: atom(),
                   Args :: [term()],
                   Time :: timeout(),
                   Ret :: term() | {error, Reason :: term()}.

start(M, F, A, Timeout) when is_atom(M), is_atom(F), is_list(A) ->
    PidRef = spawn_mon(M, F, A),
    sync_wait_mon(PidRef, Timeout).

-spec start(Module, Function, Args, Time, SpawnOpts) -> Ret
               when
                   Module :: module(),
                   Function :: atom(),
                   Args :: [term()],
                   Time :: timeout(),
                   SpawnOpts :: [spawn_option()],
                   Ret :: term() | {error, Reason :: term()}.

start(M, F, A, Timeout, SpawnOpts)
    when is_atom(M), is_atom(F), is_list(A) ->
    PidRef = spawn_opt_mon(M, F, A, SpawnOpts),
    sync_wait_mon(PidRef, Timeout).

-spec start_link(Module, Function, Args) -> Ret
                    when
                        Module :: module(),
                        Function :: atom(),
                        Args :: [term()],
                        Ret :: term() | {error, Reason :: term()}.

start_link(M, F, A) when is_atom(M), is_atom(F), is_list(A) ->
    start_link(M, F, A, infinity).

-spec start_link(Module, Function, Args, Time) -> Ret
                    when
                        Module :: module(),
                        Function :: atom(),
                        Args :: [term()],
                        Time :: timeout(),
                        Ret :: term() | {error, Reason :: term()}.

start_link(M, F, A, Timeout) when is_atom(M), is_atom(F), is_list(A) ->
    Pid = proc_lib:spawn_link(M, F, A),
    sync_wait(Pid, Timeout).

-spec start_link(Module, Function, Args, Time, SpawnOpts) -> Ret
                    when
                        Module :: module(),
                        Function :: atom(),
                        Args :: [term()],
                        Time :: timeout(),
                        SpawnOpts :: [spawn_option()],
                        Ret :: term() | {error, Reason :: term()}.

start_link(M, F, A, Timeout, SpawnOpts)
    when is_atom(M), is_atom(F), is_list(A) ->
    Pid = proc_lib:spawn_opt(M, F, A, ensure_link(SpawnOpts)),
    sync_wait(Pid, Timeout).

sync_wait(Pid, Timeout) ->
    receive
        {ack,Pid,Return} ->
            Return;
        {'EXIT',Pid,Reason} ->
            {error,Reason}
    after
        Timeout ->
            unlink(Pid),
            exit(Pid, kill),
            flush(Pid),
            {error,timeout}
    end.

sync_wait_mon({Pid,Ref}, Timeout) ->
    receive
        {ack,Pid,Return} ->
            demonitor(Ref, [flush]),
            Return;
        {'DOWN',Ref,_Type,Pid,Reason} ->
            {error,Reason};
        {'EXIT',Pid,Reason} ->
            demonitor(Ref, [flush]),
            {error,Reason}
    after
        Timeout ->
            demonitor(Ref, [flush]),
            exit(Pid, kill),
            flush(Pid),
            {error,timeout}
    end.

-spec flush(pid()) -> true.

flush(Pid) ->
    receive
        {'EXIT',Pid,_} ->
            true
    after
        0 -> true
    end.

-spec init_ack(Parent, Ret) -> ok when Parent :: pid(), Ret :: term().

init_ack(Parent, Return) ->
    Parent ! {ack,self(),Return},
    ok.

-spec init_ack(Ret) -> ok when Ret :: term().

init_ack(Return) ->
    [Parent|_] = get('$ancestors'),
    init_ack(Parent, Return).

-spec initial_call(Process) -> {Module, Function, Args} | false
                      when
                          Process :: dict_or_pid(),
                          Module :: module(),
                          Function :: atom(),
                          Args :: [atom()].

initial_call(DictOrPid) ->
    case raw_initial_call(DictOrPid) of
        {M,F,A} ->
            {M,F,make_dummy_args(A, [])};
        false ->
            false
    end.

make_dummy_args(0, Acc) ->
    Acc;
make_dummy_args(N, Acc) ->
    Arg = list_to_atom("Argument__" ++ integer_to_list(N)),
    make_dummy_args(N - 1, [Arg|Acc]).

-spec translate_initial_call(Process) -> {Module, Function, Arity}
                                when
                                    Process :: dict_or_pid(),
                                    Module :: module(),
                                    Function :: atom(),
                                    Arity :: byte().

translate_initial_call(DictOrPid) ->
    case raw_initial_call(DictOrPid) of
        {_,_,_} = MFA ->
            MFA;
        false ->
            {proc_lib,init_p,5}
    end.

raw_initial_call({X,Y,Z})
    when is_integer(X), is_integer(Y), is_integer(Z) ->
    raw_initial_call(c:pid(X, Y, Z));
raw_initial_call(Pid) when is_pid(Pid) ->
    case get_process_info(Pid, dictionary) of
        {dictionary,Dict} ->
            raw_init_call(Dict);
        _ ->
            false
    end;
raw_initial_call(ProcInfo) when is_list(ProcInfo) ->
    case lists:keyfind(dictionary, 1, ProcInfo) of
        {dictionary,Dict} ->
            raw_init_call(Dict);
        _ ->
            false
    end.

raw_init_call(Dict) ->
    case lists:keyfind('$initial_call', 1, Dict) of
        {_,{_,_,_} = MFA} ->
            MFA;
        _ ->
            false
    end.

trans_init(gen, init_it, [gen_server,_,_,supervisor,{_,Module,_},_]) ->
    {supervisor,Module,1};
trans_init(gen, init_it, [gen_server,_,_,_,supervisor,{_,Module,_},_]) ->
    {supervisor,Module,1};
trans_init(gen,
           init_it,
           [gen_server,_,_,supervisor_bridge,[Module|_],_]) ->
    {supervisor_bridge,Module,1};
trans_init(gen,
           init_it,
           [gen_server,_,_,_,supervisor_bridge,[Module|_],_]) ->
    {supervisor_bridge,Module,1};
trans_init(gen, init_it, [gen_event|_]) ->
    {gen_event,init_it,6};
trans_init(gen, init_it, [_GenMod,_,_,Module,_,_]) when is_atom(Module) ->
    {Module,init,1};
trans_init(gen, init_it, [_GenMod,_,_,_,Module|_]) when is_atom(Module) ->
    {Module,init,1};
trans_init(M, F, A) when is_atom(M), is_atom(F) ->
    {M,F,length(A)}.

crash_report(exit, normal, _, _) ->
    ok;
crash_report(exit, shutdown, _, _) ->
    ok;
crash_report(exit, {shutdown,_}, _, _) ->
    ok;
crash_report(Class, Reason, StartF, Stacktrace) ->
    case logger:allow(error, proc_lib) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {proc_lib,crash_report,4},
                     line => 508,
                     file => "proc_lib.erl"},
                   error,
                   #{label => {proc_lib,crash},
                     report =>
                         [my_info(Class, Reason, StartF, Stacktrace),
                          linked_info(self())]},
                   #{domain => [otp,sasl],
                     report_cb => fun proc_lib:report_cb/2,
                     logger_formatter => #{title => "CRASH REPORT"},
                     error_logger =>
                         #{tag => error_report,type => crash_report}}]);
        false ->
            ok
    end.

my_info(Class, Reason, [], Stacktrace) ->
    my_info_1(Class, Reason, Stacktrace);
my_info(Class, Reason, StartF, Stacktrace) ->
    [{initial_call,StartF}|my_info_1(Class, Reason, Stacktrace)].

my_info_1(Class, Reason, Stacktrace) ->
    [{pid,self()},
     get_process_info(self(), registered_name),
     {error_info,{Class,Reason,Stacktrace}},
     get_ancestors(self()),
     get_process_info(self(), message_queue_len),
     get_messages(self()),
     get_process_info(self(), links),
     get_cleaned_dictionary(self()),
     get_process_info(self(), trap_exit),
     get_process_info(self(), status),
     get_process_info(self(), heap_size),
     get_process_info(self(), stack_size),
     get_process_info(self(), reductions)].

-spec get_ancestors(pid()) -> {ancestors, [pid()]}.

get_ancestors(Pid) ->
    case get_dictionary(Pid, '$ancestors') of
        {'$ancestors',Ancestors} ->
            {ancestors,Ancestors};
        _ ->
            {ancestors,[]}
    end.

get_messages(Pid) ->
    Messages = get_process_messages(Pid),
    {messages,error_logger:limit_term(Messages)}.

get_process_messages(Pid) ->
    Depth = error_logger:get_format_depth(),
    case
        Pid =/= self()
        orelse
        Depth =:= unlimited
    of
        true ->
            {messages,Messages} = get_process_info(Pid, messages),
            Messages;
        false ->
            receive_messages(Depth)
    end.

receive_messages(0) ->
    [];
receive_messages(N) ->
    receive
        M ->
            [M|receive_messages(N - 1)]
    after
        0 -> []
    end.

get_cleaned_dictionary(Pid) ->
    case get_process_info(Pid, dictionary) of
        {dictionary,Dict} ->
            {dictionary,cleaned_dict(Dict)};
        _ ->
            {dictionary,[]}
    end.

cleaned_dict(Dict) ->
    CleanDict = clean_dict(Dict),
    error_logger:limit_term(CleanDict).

clean_dict([{'$ancestors',_}|Dict]) ->
    clean_dict(Dict);
clean_dict([{'$initial_call',_}|Dict]) ->
    clean_dict(Dict);
clean_dict([E|Dict]) ->
    [E|clean_dict(Dict)];
clean_dict([]) ->
    [].

get_dictionary(Pid, Tag) ->
    case get_process_info(Pid, dictionary) of
        {dictionary,Dict} ->
            case lists:keysearch(Tag, 1, Dict) of
                {value,Value} ->
                    Value;
                _ ->
                    undefined
            end;
        _ ->
            undefined
    end.

linked_info(Pid) ->
    make_neighbour_reports1(neighbours(Pid)).

make_neighbour_reports1([P|Ps]) ->
    ReportBody = make_neighbour_report(P),
    case lists:member(undefined, ReportBody) of
        true ->
            make_neighbour_reports1(Ps);
        false ->
            [{neighbour,ReportBody}|make_neighbour_reports1(Ps)]
    end;
make_neighbour_reports1([]) ->
    [].

make_neighbour_report(Pid) ->
    [{pid,Pid},
     get_process_info(Pid, registered_name),
     get_initial_call(Pid),
     get_process_info(Pid, current_function),
     get_ancestors(Pid),
     get_process_info(Pid, message_queue_len),
     get_process_info(Pid, links),
     get_process_info(Pid, trap_exit),
     get_process_info(Pid, status),
     get_process_info(Pid, heap_size),
     get_process_info(Pid, stack_size),
     get_process_info(Pid, reductions),
     get_process_info(Pid, current_stacktrace)].

get_initial_call(Pid) ->
    case get_dictionary(Pid, '$initial_call') of
        {'$initial_call',{M,F,A}} ->
            {initial_call,{M,F,make_dummy_args(A, [])}};
        _ ->
            get_process_info(Pid, initial_call)
    end.

-spec neighbours(pid()) -> [pid()].

neighbours(Pid) ->
    {_,Visited} = visit(adjacents(Pid), {max_neighbours(),[Pid]}),
    lists:delete(Pid, Visited).

max_neighbours() ->
    15.

visit([P|Ps], {N,Vs} = NVs) when N > 0 ->
    case lists:member(P, Vs) of
        false ->
            visit(adjacents(P), visit(Ps, {N - 1,[P|Vs]}));
        true ->
            visit(Ps, NVs)
    end;
visit(_, {_N,_Vs} = NVs) ->
    NVs.

-spec adjacents(pid()) -> [pid()].

adjacents(Pid) ->
    case catch proc_info(Pid, links) of
        {links,Links} ->
            no_trap(Links);
        _ ->
            []
    end.

no_trap([P|Ps]) ->
    case catch proc_info(P, trap_exit) of
        {trap_exit,false} ->
            [P|no_trap(Ps)];
        _ ->
            no_trap(Ps)
    end;
no_trap([]) ->
    [].

get_process_info(Pid, Tag) ->
    translate_process_info(Tag, catch proc_info(Pid, Tag)).

translate_process_info(registered_name, []) ->
    {registered_name,[]};
translate_process_info(_, {'EXIT',_}) ->
    undefined;
translate_process_info(_, Result) ->
    Result.

get_my_name() ->
    case proc_info(self(), registered_name) of
        {registered_name,Name} ->
            Name;
        _ ->
            self()
    end.

-spec get_ancestors() -> [pid()].

get_ancestors() ->
    case get('$ancestors') of
        A when is_list(A) ->
            A;
        _ ->
            []
    end.

proc_info(Pid, Item) when node(Pid) =:= node() ->
    process_info(Pid, Item);
proc_info(Pid, Item) ->
    case lists:member(node(Pid), nodes()) of
        true ->
            check(rpc:call(node(Pid), erlang, process_info, [Pid,Item]));
        _ ->
            hidden
    end.

check({badrpc,nodedown}) ->
    undefined;
check({badrpc,Error}) ->
    Error;
check(Res) ->
    Res.

-spec report_cb(CrashReport, FormatOpts) -> unicode:chardata()
                   when
                       CrashReport ::
                           #{label => {proc_lib, crash},
                             report => [term()]},
                       FormatOpts :: logger:report_cb_config().

report_cb(#{label := {proc_lib,crash},report := CrashReport}, Extra) ->
    Default =
        #{chars_limit => unlimited,
          depth => unlimited,
          single_line => false,
          encoding => utf8},
    do_format(CrashReport, maps:merge(Default, Extra)).

-spec format(CrashReport) -> string() when CrashReport :: [term()].

format(CrashReport) ->
    format(CrashReport, latin1).

-spec format(CrashReport, Encoding) -> string()
                when
                    CrashReport :: [term()],
                    Encoding :: latin1 | unicode | utf8.

format(CrashReport, Encoding) ->
    format(CrashReport, Encoding, unlimited).

-spec format(CrashReport, Encoding, Depth) -> string()
                when
                    CrashReport :: [term()],
                    Encoding :: latin1 | unicode | utf8,
                    Depth :: unlimited | pos_integer().

format(CrashReport, Encoding, Depth) ->
    do_format(CrashReport,
              #{chars_limit => unlimited,
                depth => Depth,
                encoding => Encoding,
                single_line => false}).

do_format([OwnReport,LinkReport], #{single_line := Single} = Extra) ->
    Indent =
        if
            Single ->
                "";
            true ->
                "  "
        end,
    MyIndent = Indent ++ Indent,
    Sep = nl(Single, "; "),
    OwnFormat = format_report(OwnReport, MyIndent, Extra),
    LinkFormat =
        lists:join(Sep, format_link_report(LinkReport, MyIndent, Extra)),
    Nl = nl(Single, " "),
    Str =
        io_lib:format("~scrasher:"
                      ++
                      Nl
                      ++
                      "~ts" ++ Sep ++ "~sneighbours:" ++ Nl ++ "~ts",
                      [Indent,OwnFormat,Indent,LinkFormat]),
    lists:flatten(Str).

format_link_report([Link|Reps],
                   Indent0,
                   #{single_line := Single} = Extra) ->
    Rep =
        case Link of
            {neighbour,Rep0} ->
                Rep0;
            _ ->
                Link
        end,
    Indent =
        if
            Single ->
                "";
            true ->
                Indent0
        end,
    LinkIndent = ["  ",Indent],
    [[Indent,
      "neighbour:",
      nl(Single, " "),
      format_report(Rep, LinkIndent, Extra)]|
     format_link_report(Reps, Indent, Extra)];
format_link_report(Rep, Indent, Extra) ->
    format_report(Rep, Indent, Extra).

format_report(Rep, Indent, #{single_line := Single} = Extra)
    when is_list(Rep) ->
    lists:join(nl(Single, ", "), format_rep(Rep, Indent, Extra));
format_report(Rep,
              Indent0,
              #{encoding := Enc,
                depth := Depth,
                chars_limit := Limit,
                single_line := Single}) ->
    {P,Tl} = p(Enc, Depth),
    {Indent,Width} =
        if
            Single ->
                {"","0"};
            true ->
                {Indent0,""}
        end,
    Opts =
        if
            is_integer(Limit) ->
                [{chars_limit,Limit}];
            true ->
                []
        end,
    io_lib:format("~s~" ++ Width ++ P, [Indent,Rep|Tl], Opts).

format_rep([{initial_call,InitialCall}|Rep], Indent, Extra) ->
    [format_mfa(Indent, InitialCall, Extra)|
     format_rep(Rep, Indent, Extra)];
format_rep([{error_info,{Class,Reason,StackTrace}}|Rep], Indent, Extra) ->
    [format_exception(Class, Reason, StackTrace, Extra)|
     format_rep(Rep, Indent, Extra)];
format_rep([{Tag,Data}|Rep], Indent, Extra) ->
    [format_tag(Indent, Tag, Data, Extra)|
     format_rep(Rep, Indent, Extra)];
format_rep(_, _, _Extra) ->
    [].

format_exception(Class,
                 Reason,
                 StackTrace,
                 #{encoding := Enc,
                   depth := Depth,
                   chars_limit := Limit,
                   single_line := Single} =
                     Extra) ->
    PF = pp_fun(Extra),
    StackFun =
        fun(M, _F, _A) ->
               (M =:= erl_eval) or (M =:= proc_lib)
        end,
    if
        Single ->
            {P,Tl} = p(Enc, Depth),
            Opts =
                if
                    is_integer(Limit) ->
                        [{chars_limit,Limit}];
                    true ->
                        []
                end,
            [atom_to_list(Class),
             ": ",
             io_lib:format("~0" ++ P, [{Reason,StackTrace}|Tl], Opts)];
        true ->
            EI = "    ",
            [EI,
             erl_error:format_exception(1 + length(EI),
                                        Class,
                                        Reason,
                                        StackTrace,
                                        StackFun,
                                        PF,
                                        Enc)]
    end.

format_mfa(Indent0,
           {M,F,Args} = StartF,
           #{encoding := Enc,single_line := Single} = Extra) ->
    Indent =
        if
            Single ->
                "";
            true ->
                Indent0
        end,
    try
        A = length(Args),
        [Indent,
         "initial call: ",
         atom_to_list(M),
         $:,
         to_string(F, Enc),
         $/,
         integer_to_list(A)]
    catch
        error:_ ->
            format_tag(Indent, initial_call, StartF, Extra)
    end.

to_string(A, latin1) ->
    io_lib:write_atom_as_latin1(A);
to_string(A, _) ->
    io_lib:write_atom(A).

pp_fun(#{encoding := Enc,
         depth := Depth,
         chars_limit := Limit,
         single_line := Single}) ->
    {P,Tl} = p(Enc, Depth),
    Width =
        if
            Single ->
                "0";
            true ->
                ""
        end,
    Opts =
        if
            is_integer(Limit) ->
                [{chars_limit,Limit}];
            true ->
                []
        end,
    fun(Term, I) ->
           io_lib:format("~" ++ Width ++ "." ++ integer_to_list(I) ++ P,
                         [Term|Tl],
                         Opts)
    end.

format_tag(Indent0,
           Tag,
           Data,
           #{encoding := Enc,
             depth := Depth,
             chars_limit := Limit,
             single_line := Single}) ->
    {P,Tl} = p(Enc, Depth),
    {Indent,Width} =
        if
            Single ->
                {"","0"};
            true ->
                {Indent0,""}
        end,
    Opts =
        if
            is_integer(Limit) ->
                [{chars_limit,Limit}];
            true ->
                []
        end,
    io_lib:format("~s~" ++ Width ++ "p: ~" ++ Width ++ ".18" ++ P,
                  [Indent,Tag,Data|Tl],
                  Opts).

p(Encoding, Depth) ->
    {Letter,Tl} =
        case Depth of
            unlimited ->
                {"p",[]};
            _ ->
                {"P",[Depth]}
        end,
    P = modifier(Encoding) ++ Letter,
    {P,Tl}.

modifier(latin1) ->
    "";
modifier(_) ->
    "t".

nl(true, Else) ->
    Else;
nl(false, _) ->
    "\n".

-spec stop(Process) -> ok
              when
                  Process :: pid() | RegName | {RegName, node()},
                  RegName :: atom().

stop(Process) ->
    stop(Process, normal, infinity).

-spec stop(Process, Reason, Timeout) -> ok
              when
                  Process :: pid() | RegName | {RegName, node()},
                  RegName :: atom(),
                  Reason :: term(),
                  Timeout :: timeout().

stop(Process, Reason, Timeout) ->
    {Pid,Mref} = spawn_monitor(do_stop(Process, Reason)),
    receive
        {'DOWN',Mref,_,_,Reason} ->
            ok;
        {'DOWN',Mref,_,_,{noproc,{sys,terminate,_}}} ->
            exit(noproc);
        {'DOWN',Mref,_,_,CrashReason} ->
            exit(CrashReason)
    after
        Timeout ->
            exit(Pid, kill),
            receive
                {'DOWN',Mref,_,_,_} ->
                    exit(timeout)
            end
    end.

-spec do_stop(Process, Reason) -> Fun
                 when
                     Process :: pid() | RegName | {RegName, node()},
                     RegName :: atom(),
                     Reason :: term(),
                     Fun :: fun(() -> no_return()).

do_stop(Process, Reason) ->
    fun() ->
           Mref = monitor(process, Process),
           ok = sys:terminate(Process, Reason, infinity),
           receive
               {'DOWN',Mref,_,_,ExitReason} ->
                   exit(ExitReason)
           end
    end.



