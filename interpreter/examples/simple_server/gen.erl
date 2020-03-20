-file("gen.erl", 1).

-module(gen).

-compile({inline,[{get_node,1}]}).

-export([start/5,
         start/6,
         debug_options/2,
         hibernate_after/1,
         name/1,
         unregister_name/1,
         get_proc_name/1,
         get_parent/0,
         call/3,
         call/4,
         reply/2,
         stop/1,
         stop/3]).

-export([init_it/6,init_it/7]).

-export([format_status_header/2]).

-type linkage() :: link | nolink.

-type emgr_name() ::
          {local, atom()} |
          {global, term()} |
          {via, Module :: module(), Name :: term()}.

-type start_ret() :: {ok, pid()} | ignore | {error, term()}.

-type debug_flag() ::
          trace | log | statistics | debug | {logfile, string()}.

-type option() ::
          {timeout, timeout()} |
          {debug, [debug_flag()]} |
          {hibernate_after, timeout()} |
          {spawn_opt, [proc_lib:spawn_option()]}.

-type options() :: [option()].

-spec start(module(),
            linkage(),
            emgr_name(),
            module(),
            term(),
            options()) ->
               start_ret().

start(GenMod, LinkP, Name, Mod, Args, Options) ->
    case where(Name) of
        undefined ->
            do_spawn(GenMod, LinkP, Name, Mod, Args, Options);
        Pid ->
            {error,{already_started,Pid}}
    end.

-spec start(module(), linkage(), module(), term(), options()) ->
               start_ret().

start(GenMod, LinkP, Mod, Args, Options) ->
    do_spawn(GenMod, LinkP, Mod, Args, Options).

do_spawn(GenMod, link, Mod, Args, Options) ->
    Time = timeout(Options),
    proc_lib:start_link(gen,
                        init_it,
                        [GenMod,self(),self(),Mod,Args,Options],
                        Time,
                        spawn_opts(Options));
do_spawn(GenMod, _, Mod, Args, Options) ->
    Time = timeout(Options),
    proc_lib:start(gen,
                   init_it,
                   [GenMod,self(),self,Mod,Args,Options],
                   Time,
                   spawn_opts(Options)).

do_spawn(GenMod, link, Name, Mod, Args, Options) ->
    Time = timeout(Options),
    proc_lib:start_link(gen,
                        init_it,
                        [GenMod,self(),self(),Name,Mod,Args,Options],
                        Time,
                        spawn_opts(Options));
do_spawn(GenMod, _, Name, Mod, Args, Options) ->
    Time = timeout(Options),
    proc_lib:start(gen,
                   init_it,
                   [GenMod,self(),self,Name,Mod,Args,Options],
                   Time,
                   spawn_opts(Options)).

init_it(GenMod, Starter, Parent, Mod, Args, Options) ->
    init_it2(GenMod, Starter, Parent, self(), Mod, Args, Options).

init_it(GenMod, Starter, Parent, Name, Mod, Args, Options) ->
    case register_name(Name) of
        true ->
            init_it2(GenMod, Starter, Parent, Name, Mod, Args, Options);
        {false,Pid} ->
            proc_lib:init_ack(Starter, {error,{already_started,Pid}})
    end.

init_it2(GenMod, Starter, Parent, Name, Mod, Args, Options) ->
    GenMod:init_it(Starter, Parent, Name, Mod, Args, Options).

call(Process, Label, Request) ->
    call(Process, Label, Request, 5000).

call(Process, Label, Request, Timeout)
    when
        is_pid(Process),
        Timeout =:= infinity
        orelse
        is_integer(Timeout)
        andalso
        Timeout >= 0 ->
    do_call(Process, Label, Request, Timeout);
call(Process, Label, Request, Timeout)
    when Timeout =:= infinity; is_integer(Timeout), Timeout >= 0 ->
    Fun =
        fun(Pid) ->
               do_call(Pid, Label, Request, Timeout)
        end,
    do_for_proc(Process, Fun).

do_call(Process, Label, Request, Timeout)
    when is_atom(Process) =:= false ->
    Mref = monitor(process, Process),
    erlang:send(Process, {Label,{self(),Mref},Request}, [noconnect]),
    receive
        {Mref,Reply} ->
            demonitor(Mref, [flush]),
            {ok,Reply};
        {'DOWN',Mref,_,_,noconnection} ->
            Node = get_node(Process),
            exit({nodedown,Node});
        {'DOWN',Mref,_,_,Reason} ->
            exit(Reason)
    after
        Timeout ->
            demonitor(Mref, [flush]),
            exit(timeout)
    end.

get_node(Process) ->
    case Process of
        {_S,N} when is_atom(N) ->
            N;
        _ when is_pid(Process) ->
            node(Process)
    end.

reply({To,Tag}, Reply) ->
    Msg = {Tag,Reply},
    try
        To ! Msg
    catch
        _:_ ->
            Msg
    end.

stop(Process) ->
    stop(Process, normal, infinity).

stop(Process, Reason, Timeout)
    when Timeout =:= infinity; is_integer(Timeout), Timeout >= 0 ->
    Fun =
        fun(Pid) ->
               proc_lib:stop(Pid, Reason, Timeout)
        end,
    do_for_proc(Process, Fun).

do_for_proc(Pid, Fun) when is_pid(Pid) ->
    Fun(Pid);
do_for_proc(Name, Fun) when is_atom(Name) ->
    case whereis(Name) of
        Pid when is_pid(Pid) ->
            Fun(Pid);
        undefined ->
            exit(noproc)
    end;
do_for_proc(Process, Fun)
    when
        tuple_size(Process) == 2
        andalso
        element(1, Process) == global
        orelse
        tuple_size(Process) == 3
        andalso
        element(1, Process) == via ->
    case where(Process) of
        Pid when is_pid(Pid) ->
            Node = node(Pid),
            try
                Fun(Pid)
            catch
                exit:{nodedown,Node} ->
                    exit(noproc)
            end;
        undefined ->
            exit(noproc)
    end;
do_for_proc({Name,Node}, Fun) when Node =:= node() ->
    do_for_proc(Name, Fun);
do_for_proc({_Name,Node} = Process, Fun) when is_atom(Node) ->
    if
        node() =:= nonode@nohost ->
            exit({nodedown,Node});
        true ->
            Fun(Process)
    end.

where({global,Name}) ->
    global:whereis_name(Name);
where({via,Module,Name}) ->
    Module:whereis_name(Name);
where({local,Name}) ->
    whereis(Name).

register_name({local,Name} = LN) ->
    try register(Name, self()) of
        true ->
            true
    catch
        error:_ ->
            {false,where(LN)}
    end;
register_name({global,Name} = GN) ->
    case global:register_name(Name, self()) of
        yes ->
            true;
        no ->
            {false,where(GN)}
    end;
register_name({via,Module,Name} = GN) ->
    case Module:register_name(Name, self()) of
        yes ->
            true;
        no ->
            {false,where(GN)}
    end.

name({local,Name}) ->
    Name;
name({global,Name}) ->
    Name;
name({via,_,Name}) ->
    Name;
name(Pid) when is_pid(Pid) ->
    Pid.

unregister_name({local,Name}) ->
    try unregister(Name) of
        _ ->
            ok
    catch
        _:_ ->
            ok
    end;
unregister_name({global,Name}) ->
    _ = global:unregister_name(Name),
    ok;
unregister_name({via,Mod,Name}) ->
    _ = Mod:unregister_name(Name),
    ok;
unregister_name(Pid) when is_pid(Pid) ->
    ok.

get_proc_name(Pid) when is_pid(Pid) ->
    Pid;
get_proc_name({local,Name}) ->
    case process_info(self(), registered_name) of
        {registered_name,Name} ->
            Name;
        {registered_name,_Name} ->
            exit(process_not_registered);
        [] ->
            exit(process_not_registered)
    end;
get_proc_name({global,Name}) ->
    case global:whereis_name(Name) of
        undefined ->
            exit(process_not_registered_globally);
        Pid when Pid =:= self() ->
            Name;
        _Pid ->
            exit(process_not_registered_globally)
    end;
get_proc_name({via,Mod,Name}) ->
    case Mod:whereis_name(Name) of
        undefined ->
            exit({process_not_registered_via,Mod});
        Pid when Pid =:= self() ->
            Name;
        _Pid ->
            exit({process_not_registered_via,Mod})
    end.

get_parent() ->
    case get('$ancestors') of
        [Parent|_] when is_pid(Parent) ->
            Parent;
        [Parent|_] when is_atom(Parent) ->
            name_to_pid(Parent);
        _ ->
            exit(process_was_not_started_by_proc_lib)
    end.

name_to_pid(Name) ->
    case whereis(Name) of
        undefined ->
            case global:whereis_name(Name) of
                undefined ->
                    exit(could_not_find_registered_name);
                Pid ->
                    Pid
            end;
        Pid ->
            Pid
    end.

timeout(Options) ->
    case lists:keyfind(timeout, 1, Options) of
        {_,Time} ->
            Time;
        false ->
            infinity
    end.

spawn_opts(Options) ->
    case lists:keyfind(spawn_opt, 1, Options) of
        {_,Opts} ->
            Opts;
        false ->
            []
    end.

hibernate_after(Options) ->
    case lists:keyfind(hibernate_after, 1, Options) of
        {_,HibernateAfterTimeout} ->
            HibernateAfterTimeout;
        false ->
            infinity
    end.

debug_options(Name, Opts) ->
    case lists:keyfind(debug, 1, Opts) of
        {_,Options} ->
            try
                sys:debug_options(Options)
            catch
                _:_ ->
                    error_logger:format("~tp: ignoring erroneous debug "
                                        "options - ~tp~n",
                                        [Name,Options]),
                    []
            end;
        false ->
            []
    end.

format_status_header(TagLine, Pid) when is_pid(Pid) ->
    lists:concat([TagLine," ",pid_to_list(Pid)]);
format_status_header(TagLine, RegName) when is_atom(RegName) ->
    lists:concat([TagLine," ",RegName]);
format_status_header(TagLine, Name) ->
    {TagLine,Name}.



