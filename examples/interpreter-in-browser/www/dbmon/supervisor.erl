-file("supervisor.erl", 1).

-module(supervisor).

-behaviour(gen_server).

-export([start_link/2,
         start_link/3,
         start_child/2,
         restart_child/2,
         delete_child/2,
         terminate_child/2,
         which_children/1,
         count_children/1,
         check_childspecs/1,
         get_childspec/2]).

-export([init/1,
         handle_call/3,
         handle_cast/2,
         handle_info/2,
         terminate/2,
         code_change/3,
         format_status/2]).

-export([get_callback_module/1]).

-file("/home/hansihe/.asdf/plugins/erlang/kerl-home/builds/asdf_21.2.4/"
      "otp_src_21.2.4/lib/stdlib/src/../../kernel/include/logger.hrl",
      1).

-file("supervisor.erl", 39).

-export_type([sup_flags/0,child_spec/0,startchild_ret/0,strategy/0]).

-type child() :: undefined | pid().

-type child_id() :: term().

-type mfargs() ::
          {M :: module(), F :: atom(), A :: [term()] | undefined}.

-type modules() :: [module()] | dynamic.

-type restart() :: permanent | transient | temporary.

-type shutdown() :: brutal_kill | timeout().

-type worker() :: worker | supervisor.

-type sup_name() ::
          {local, Name :: atom()} |
          {global, Name :: atom()} |
          {via, Module :: module(), Name :: any()}.

-type sup_ref() ::
          (Name :: atom()) |
          {Name :: atom(), Node :: node()} |
          {global, Name :: atom()} |
          {via, Module :: module(), Name :: any()} |
          pid().

-type child_spec() ::
          #{id := child_id(),
            start := mfargs(),
            restart => restart(),
            shutdown => shutdown(),
            type => worker(),
            modules => modules()} |
          {Id :: child_id(),
           StartFunc :: mfargs(),
           Restart :: restart(),
           Shutdown :: shutdown(),
           Type :: worker(),
           Modules :: modules()}.

-type strategy() ::
          one_for_all | one_for_one | rest_for_one | simple_one_for_one.

-type sup_flags() ::
          #{strategy => strategy(),
            intensity => non_neg_integer(),
            period => pos_integer()} |
          {RestartStrategy :: strategy(),
           Intensity :: non_neg_integer(),
           Period :: pos_integer()}.

-type children() ::
          {Ids :: [child_id()], Db :: #{child_id() => child_rec()}}.

-record(child,{pid =
                   undefined ::
                       child() |
                       {restarting, pid() | undefined} |
                       [pid()],
               id :: child_id(),
               mfargs :: mfargs(),
               restart_type :: restart(),
               shutdown :: shutdown(),
               child_type :: worker(),
               modules = [] :: modules()}).

-type child_rec() :: #child{}.

-record(state,{name,
               strategy :: strategy() | undefined,
               children = {[],#{}} :: children(),
               dynamics ::
                   {maps, #{pid() => list()}} |
                   {sets, sets:set(pid())} |
                   undefined,
               intensity :: non_neg_integer() | undefined,
               period :: pos_integer() | undefined,
               restarts = [],
               dynamic_restarts = 0 :: non_neg_integer(),
               module,
               args}).

-type state() :: #state{}.

-callback init(Args :: term()) ->
                  {ok,
                   {SupFlags :: sup_flags(),
                    [ChildSpec :: child_spec()]}} |
                  ignore.

-type startlink_err() ::
          {already_started, pid()} | {shutdown, term()} | term().

-type startlink_ret() :: {ok, pid()} | ignore | {error, startlink_err()}.

-spec start_link(Module, Args) -> startlink_ret()
                    when Module :: module(), Args :: term().

start_link(Mod, Args) ->
    gen_server:start_link(supervisor, {self,Mod,Args}, []).

-spec start_link(SupName, Module, Args) -> startlink_ret()
                    when
                        SupName :: sup_name(),
                        Module :: module(),
                        Args :: term().

start_link(SupName, Mod, Args) ->
    gen_server:start_link(SupName, supervisor, {SupName,Mod,Args}, []).

-type startchild_err() ::
          already_present | {already_started, Child :: child()} | term().

-type startchild_ret() ::
          {ok, Child :: child()} |
          {ok, Child :: child(), Info :: term()} |
          {error, startchild_err()}.

-spec start_child(SupRef, ChildSpec) -> startchild_ret()
                     when
                         SupRef :: sup_ref(),
                         ChildSpec :: child_spec() | (List :: [term()]).

start_child(Supervisor, ChildSpec) ->
    call(Supervisor, {start_child,ChildSpec}).

-spec restart_child(SupRef, Id) -> Result
                       when
                           SupRef :: sup_ref(),
                           Id :: child_id(),
                           Result ::
                               {ok, Child :: child()} |
                               {ok, Child :: child(), Info :: term()} |
                               {error, Error},
                           Error ::
                               running |
                               restarting |
                               not_found |
                               simple_one_for_one |
                               term().

restart_child(Supervisor, Id) ->
    call(Supervisor, {restart_child,Id}).

-spec delete_child(SupRef, Id) -> Result
                      when
                          SupRef :: sup_ref(),
                          Id :: child_id(),
                          Result :: ok | {error, Error},
                          Error ::
                              running |
                              restarting |
                              not_found |
                              simple_one_for_one.

delete_child(Supervisor, Id) ->
    call(Supervisor, {delete_child,Id}).

-spec terminate_child(SupRef, Id) -> Result
                         when
                             SupRef :: sup_ref(),
                             Id :: pid() | child_id(),
                             Result :: ok | {error, Error},
                             Error :: not_found | simple_one_for_one.

terminate_child(Supervisor, Id) ->
    call(Supervisor, {terminate_child,Id}).

-spec get_childspec(SupRef, Id) -> Result
                       when
                           SupRef :: sup_ref(),
                           Id :: pid() | child_id(),
                           Result :: {ok, child_spec()} | {error, Error},
                           Error :: not_found.

get_childspec(Supervisor, Id) ->
    call(Supervisor, {get_childspec,Id}).

-spec which_children(SupRef) -> [{Id, Child, Type, Modules}]
                        when
                            SupRef :: sup_ref(),
                            Id :: child_id() | undefined,
                            Child :: child() | restarting,
                            Type :: worker(),
                            Modules :: modules().

which_children(Supervisor) ->
    call(Supervisor, which_children).

-spec count_children(SupRef) -> PropListOfCounts
                        when
                            SupRef :: sup_ref(),
                            PropListOfCounts :: [Count],
                            Count ::
                                {specs,
                                 ChildSpecCount :: non_neg_integer()} |
                                {active,
                                 ActiveProcessCount :: non_neg_integer()} |
                                {supervisors,
                                 ChildSupervisorCount ::
                                     non_neg_integer()} |
                                {workers,
                                 ChildWorkerCount :: non_neg_integer()}.

count_children(Supervisor) ->
    call(Supervisor, count_children).

call(Supervisor, Req) ->
    gen_server:call(Supervisor, Req, infinity).

-spec check_childspecs(ChildSpecs) -> Result
                          when
                              ChildSpecs :: [child_spec()],
                              Result :: ok | {error, Error :: term()}.

check_childspecs(ChildSpecs) when is_list(ChildSpecs) ->
    case check_startspec(ChildSpecs) of
        {ok,_} ->
            ok;
        Error ->
            {error,Error}
    end;
check_childspecs(X) ->
    {error,{badarg,X}}.

-spec get_callback_module(Pid) -> Module
                             when Pid :: pid(), Module :: atom().

get_callback_module(Pid) ->
    {status,_Pid,{module,_Mod},[_PDict,_SysState,_Parent,_Dbg,Misc]} =
        sys:get_status(Pid),
    case lists:keyfind(supervisor, 1, Misc) of
        {supervisor,[{"Callback",Mod}]} ->
            Mod;
        _ ->
            [_Header,_Data,{data,[{"State",State}]}|_] = Misc,
            State#state.module
    end.

-type init_sup_name() :: sup_name() | self.

-type stop_rsn() ::
          {shutdown, term()} |
          {bad_return, {module(), init, term()}} |
          {bad_start_spec, term()} |
          {start_spec, term()} |
          {supervisor_data, term()}.

-spec init({init_sup_name(), module(), [term()]}) ->
              {ok, state()} | ignore | {stop, stop_rsn()}.

init({SupName,Mod,Args}) ->
    process_flag(trap_exit, true),
    case Mod:init(Args) of
        {ok,{SupFlags,StartSpec}} ->
            case init_state(SupName, SupFlags, Mod, Args) of
                {ok,State}
                    when State#state.strategy =:= simple_one_for_one ->
                    init_dynamic(State, StartSpec);
                {ok,State} ->
                    init_children(State, StartSpec);
                Error ->
                    {stop,{supervisor_data,Error}}
            end;
        ignore ->
            ignore;
        Error ->
            {stop,{bad_return,{Mod,init,Error}}}
    end.

init_children(State, StartSpec) ->
    SupName = State#state.name,
    case check_startspec(StartSpec) of
        {ok,Children} ->
            case start_children(Children, SupName) of
                {ok,NChildren} ->
                    {ok,State#state{children = NChildren}};
                {error,NChildren,Reason} ->
                    _ = terminate_children(NChildren, SupName),
                    {stop,{shutdown,Reason}}
            end;
        Error ->
            {stop,{start_spec,Error}}
    end.

init_dynamic(State, [StartSpec]) ->
    case check_startspec([StartSpec]) of
        {ok,Children} ->
            {ok,dyn_init(State#state{children = Children})};
        Error ->
            {stop,{start_spec,Error}}
    end;
init_dynamic(_State, StartSpec) ->
    {stop,{bad_start_spec,StartSpec}}.

start_children(Children, SupName) ->
    Start =
        fun(Id, Child) ->
               case do_start_child(SupName, Child) of
                   {ok,undefined}
                       when Child#child.restart_type =:= temporary ->
                       remove;
                   {ok,Pid} ->
                       {update,Child#child{pid = Pid}};
                   {ok,Pid,_Extra} ->
                       {update,Child#child{pid = Pid}};
                   {error,Reason} ->
                       case logger:allow(error, supervisor) of
                           true ->
                               apply(logger,
                                     macro_log,
                                     [#{mfa =>
                                            {supervisor,
                                             start_children,
                                             2},
                                        line => 357,
                                        file => "supervisor.erl"},
                                      error,
                                      #{label =>
                                            {supervisor,start_error},
                                        report =>
                                            [{supervisor,SupName},
                                             {errorContext,start_error},
                                             {reason,Reason},
                                             {offender,
                                              extract_child(Child)}]},
                                      #{domain => [otp,sasl],
                                        report_cb =>
                                            fun logger:format_otp_report/1,
                                        logger_formatter =>
                                            #{title =>
                                                  "SUPERVISOR REPORT"},
                                        error_logger =>
                                            #{tag => error_report,
                                              type => supervisor_report}}]);
                           false ->
                               ok
                       end,
                       {abort,{failed_to_start_child,Id,Reason}}
               end
        end,
    children_map(Start, Children).

do_start_child(SupName, Child) ->
    #child{mfargs = {M,F,Args}} = Child,
    case do_start_child_i(M, F, Args) of
        {ok,Pid} when is_pid(Pid) ->
            NChild = Child#child{pid = Pid},
            report_progress(NChild, SupName),
            {ok,Pid};
        {ok,Pid,Extra} when is_pid(Pid) ->
            NChild = Child#child{pid = Pid},
            report_progress(NChild, SupName),
            {ok,Pid,Extra};
        Other ->
            Other
    end.

do_start_child_i(M, F, A) ->
    case catch apply(M, F, A) of
        {ok,Pid} when is_pid(Pid) ->
            {ok,Pid};
        {ok,Pid,Extra} when is_pid(Pid) ->
            {ok,Pid,Extra};
        ignore ->
            {ok,undefined};
        {error,Error} ->
            {error,Error};
        What ->
            {error,What}
    end.

-type call() :: which_children | count_children | {_, _}.

-spec handle_call(call(), term(), state()) -> {reply, term(), state()}.

handle_call({start_child,EArgs}, _From, State)
    when State#state.strategy =:= simple_one_for_one ->
    Child = get_dynamic_child(State),
    #child{mfargs = {M,F,A}} = Child,
    Args = A ++ EArgs,
    case do_start_child_i(M, F, Args) of
        {ok,undefined} ->
            {reply,{ok,undefined},State};
        {ok,Pid} ->
            NState = dyn_store(Pid, Args, State),
            {reply,{ok,Pid},NState};
        {ok,Pid,Extra} ->
            NState = dyn_store(Pid, Args, State),
            {reply,{ok,Pid,Extra},NState};
        What ->
            {reply,What,State}
    end;
handle_call({start_child,ChildSpec}, _From, State) ->
    case check_childspec(ChildSpec) of
        {ok,Child} ->
            {Resp,NState} = handle_start_child(Child, State),
            {reply,Resp,NState};
        What ->
            {reply,{error,What},State}
    end;
handle_call({terminate_child,Id}, _From, State)
    when not is_pid(Id), State#state.strategy =:= simple_one_for_one ->
    {reply,{error,simple_one_for_one},State};
handle_call({terminate_child,Id}, _From, State) ->
    case find_child(Id, State) of
        {ok,Child} ->
            do_terminate(Child, State#state.name),
            {reply,ok,del_child(Child, State)};
        error ->
            {reply,{error,not_found},State}
    end;
handle_call({restart_child,_Id}, _From, State)
    when State#state.strategy =:= simple_one_for_one ->
    {reply,{error,simple_one_for_one},State};
handle_call({restart_child,Id}, _From, State) ->
    case find_child(Id, State) of
        {ok,Child} when Child#child.pid =:= undefined ->
            case do_start_child(State#state.name, Child) of
                {ok,Pid} ->
                    NState = set_pid(Pid, Id, State),
                    {reply,{ok,Pid},NState};
                {ok,Pid,Extra} ->
                    NState = set_pid(Pid, Id, State),
                    {reply,{ok,Pid,Extra},NState};
                Error ->
                    {reply,Error,State}
            end;
        {ok,#child{pid = {restarting,_}}} ->
            {reply,{error,restarting},State};
        {ok,_} ->
            {reply,{error,running},State};
        _ ->
            {reply,{error,not_found},State}
    end;
handle_call({delete_child,_Id}, _From, State)
    when State#state.strategy =:= simple_one_for_one ->
    {reply,{error,simple_one_for_one},State};
handle_call({delete_child,Id}, _From, State) ->
    case find_child(Id, State) of
        {ok,Child} when Child#child.pid =:= undefined ->
            NState = remove_child(Id, State),
            {reply,ok,NState};
        {ok,#child{pid = {restarting,_}}} ->
            {reply,{error,restarting},State};
        {ok,_} ->
            {reply,{error,running},State};
        _ ->
            {reply,{error,not_found},State}
    end;
handle_call({get_childspec,Id}, _From, State) ->
    case find_child(Id, State) of
        {ok,Child} ->
            {reply,{ok,child_to_spec(Child)},State};
        error ->
            {reply,{error,not_found},State}
    end;
handle_call(which_children, _From, State)
    when State#state.strategy =:= simple_one_for_one ->
    #child{child_type = CT,modules = Mods} = get_dynamic_child(State),
    Reply =
        dyn_map(fun({restarting,_}) ->
                       {undefined,restarting,CT,Mods};
                   (Pid) ->
                       {undefined,Pid,CT,Mods}
                end,
                State),
    {reply,Reply,State};
handle_call(which_children, _From, State) ->
    Resp =
        children_to_list(fun(Id,
                             #child{pid = {restarting,_},
                                    child_type = ChildType,
                                    modules = Mods}) ->
                                {Id,restarting,ChildType,Mods};
                            (Id,
                             #child{pid = Pid,
                                    child_type = ChildType,
                                    modules = Mods}) ->
                                {Id,Pid,ChildType,Mods}
                         end,
                         State#state.children),
    {reply,Resp,State};
handle_call(count_children,
            _From,
            #state{dynamic_restarts = Restarts} = State)
    when State#state.strategy =:= simple_one_for_one ->
    #child{child_type = CT} = get_dynamic_child(State),
    Sz = dyn_size(State),
    Active = Sz - Restarts,
    Reply =
        case CT of
            supervisor ->
                [{specs,1},{active,Active},{supervisors,Sz},{workers,0}];
            worker ->
                [{specs,1},{active,Active},{supervisors,0},{workers,Sz}]
        end,
    {reply,Reply,State};
handle_call(count_children, _From, State) ->
    {Specs,Active,Supers,Workers} =
        children_fold(fun(_Id, Child, Counts) ->
                             count_child(Child, Counts)
                      end,
                      {0,0,0,0},
                      State#state.children),
    Reply =
        [{specs,Specs},
         {active,Active},
         {supervisors,Supers},
         {workers,Workers}],
    {reply,Reply,State}.

count_child(#child{pid = Pid,child_type = worker},
            {Specs,Active,Supers,Workers}) ->
    case
        is_pid(Pid)
        andalso
        is_process_alive(Pid)
    of
        true ->
            {Specs + 1,Active + 1,Supers,Workers + 1};
        false ->
            {Specs + 1,Active,Supers,Workers + 1}
    end;
count_child(#child{pid = Pid,child_type = supervisor},
            {Specs,Active,Supers,Workers}) ->
    case
        is_pid(Pid)
        andalso
        is_process_alive(Pid)
    of
        true ->
            {Specs + 1,Active + 1,Supers + 1,Workers};
        false ->
            {Specs + 1,Active,Supers + 1,Workers}
    end.

-spec handle_cast({try_again_restart, child_id() | {restarting, pid()}},
                  state()) ->
                     {noreply, state()} | {stop, shutdown, state()}.

handle_cast({try_again_restart,TryAgainId}, State) ->
    case find_child_and_args(TryAgainId, State) of
        {ok,Child = #child{pid = {restarting,_}}} ->
            case restart(Child, State) of
                {ok,State1} ->
                    {noreply,State1};
                {shutdown,State1} ->
                    {stop,shutdown,State1}
            end;
        _ ->
            {noreply,State}
    end.

-spec handle_info(term(), state()) ->
                     {noreply, state()} | {stop, shutdown, state()}.

handle_info({'EXIT',Pid,Reason}, State) ->
    case restart_child(Pid, Reason, State) of
        {ok,State1} ->
            {noreply,State1};
        {shutdown,State1} ->
            {stop,shutdown,State1}
    end;
handle_info(Msg, State) ->
    case logger:allow(error, supervisor) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {supervisor,handle_info,2},
                     line => 582,
                     file => "supervisor.erl"},
                   error,
                   "Supervisor received unexpected message: ~tp~n",
                   [Msg],
                   #{domain => [otp],error_logger => #{tag => error}}]);
        false ->
            ok
    end,
    {noreply,State}.

-spec terminate(term(), state()) -> ok.

terminate(_Reason, State)
    when State#state.strategy =:= simple_one_for_one ->
    terminate_dynamic_children(State);
terminate(_Reason, State) ->
    terminate_children(State#state.children, State#state.name).

-spec code_change(term(), state(), term()) ->
                     {ok, state()} | {error, term()}.

code_change(_, State, _) ->
    case (State#state.module):init(State#state.args) of
        {ok,{SupFlags,StartSpec}} ->
            case set_flags(SupFlags, State) of
                {ok,State1} ->
                    update_childspec(State1, StartSpec);
                {invalid_type,SupFlags} ->
                    {error,{bad_flags,SupFlags}};
                Error ->
                    {error,Error}
            end;
        ignore ->
            {ok,State};
        Error ->
            Error
    end.

update_childspec(State, StartSpec)
    when State#state.strategy =:= simple_one_for_one ->
    case check_startspec(StartSpec) of
        {ok,{[_],_} = Children} ->
            {ok,State#state{children = Children}};
        Error ->
            {error,Error}
    end;
update_childspec(State, StartSpec) ->
    case check_startspec(StartSpec) of
        {ok,Children} ->
            OldC = State#state.children,
            NewC = update_childspec1(OldC, Children, []),
            {ok,State#state{children = NewC}};
        Error ->
            {error,Error}
    end.

update_childspec1({[Id|OldIds],OldDb}, {Ids,Db}, KeepOld) ->
    case update_chsp(maps:get(Id, OldDb), Db) of
        {ok,NewDb} ->
            update_childspec1({OldIds,OldDb}, {Ids,NewDb}, KeepOld);
        false ->
            update_childspec1({OldIds,OldDb}, {Ids,Db}, [Id|KeepOld])
    end;
update_childspec1({[],OldDb}, {Ids,Db}, KeepOld) ->
    KeepOldDb = maps:with(KeepOld, OldDb),
    {lists:reverse(Ids ++ KeepOld),maps:merge(KeepOldDb, Db)}.

update_chsp(#child{id = Id} = OldChild, NewDb) ->
    case maps:find(Id, NewDb) of
        {ok,Child} ->
            {ok,NewDb#{Id => Child#child{pid = OldChild#child.pid}}};
        error ->
            false
    end.

handle_start_child(Child, State) ->
    case find_child(Child#child.id, State) of
        error ->
            case do_start_child(State#state.name, Child) of
                {ok,undefined}
                    when Child#child.restart_type =:= temporary ->
                    {{ok,undefined},State};
                {ok,Pid} ->
                    {{ok,Pid},save_child(Child#child{pid = Pid}, State)};
                {ok,Pid,Extra} ->
                    {{ok,Pid,Extra},
                     save_child(Child#child{pid = Pid}, State)};
                {error,What} ->
                    {{error,{What,Child}},State}
            end;
        {ok,OldChild} when is_pid(OldChild#child.pid) ->
            {{error,{already_started,OldChild#child.pid}},State};
        {ok,_OldChild} ->
            {{error,already_present},State}
    end.

restart_child(Pid, Reason, State) ->
    case find_child_and_args(Pid, State) of
        {ok,Child} ->
            do_restart(Reason, Child, State);
        error ->
            {ok,State}
    end.

do_restart(Reason, Child, State)
    when Child#child.restart_type =:= permanent ->
    case logger:allow(error, supervisor) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {supervisor,do_restart,3},
                     line => 701,
                     file => "supervisor.erl"},
                   error,
                   #{label => {supervisor,child_terminated},
                     report =>
                         [{supervisor,State#state.name},
                          {errorContext,child_terminated},
                          {reason,Reason},
                          {offender,extract_child(Child)}]},
                   #{domain => [otp,sasl],
                     report_cb => fun logger:format_otp_report/1,
                     logger_formatter => #{title => "SUPERVISOR REPORT"},
                     error_logger =>
                         #{tag => error_report,
                           type => supervisor_report}}]);
        false ->
            ok
    end,
    restart(Child, State);
do_restart(normal, Child, State) ->
    NState = del_child(Child, State),
    {ok,NState};
do_restart(shutdown, Child, State) ->
    NState = del_child(Child, State),
    {ok,NState};
do_restart({shutdown,_Term}, Child, State) ->
    NState = del_child(Child, State),
    {ok,NState};
do_restart(Reason, Child, State)
    when Child#child.restart_type =:= transient ->
    case logger:allow(error, supervisor) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {supervisor,do_restart,3},
                     line => 713,
                     file => "supervisor.erl"},
                   error,
                   #{label => {supervisor,child_terminated},
                     report =>
                         [{supervisor,State#state.name},
                          {errorContext,child_terminated},
                          {reason,Reason},
                          {offender,extract_child(Child)}]},
                   #{domain => [otp,sasl],
                     report_cb => fun logger:format_otp_report/1,
                     logger_formatter => #{title => "SUPERVISOR REPORT"},
                     error_logger =>
                         #{tag => error_report,
                           type => supervisor_report}}]);
        false ->
            ok
    end,
    restart(Child, State);
do_restart(Reason, Child, State)
    when Child#child.restart_type =:= temporary ->
    case logger:allow(error, supervisor) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {supervisor,do_restart,3},
                     line => 716,
                     file => "supervisor.erl"},
                   error,
                   #{label => {supervisor,child_terminated},
                     report =>
                         [{supervisor,State#state.name},
                          {errorContext,child_terminated},
                          {reason,Reason},
                          {offender,extract_child(Child)}]},
                   #{domain => [otp,sasl],
                     report_cb => fun logger:format_otp_report/1,
                     logger_formatter => #{title => "SUPERVISOR REPORT"},
                     error_logger =>
                         #{tag => error_report,
                           type => supervisor_report}}]);
        false ->
            ok
    end,
    NState = del_child(Child, State),
    {ok,NState}.

restart(Child, State) ->
    case add_restart(State) of
        {ok,NState} ->
            case restart(NState#state.strategy, Child, NState) of
                {{try_again,TryAgainId},NState2} ->
                    try_again_restart(TryAgainId),
                    {ok,NState2};
                Other ->
                    Other
            end;
        {terminate,NState} ->
            case logger:allow(error, supervisor) of
                true ->
                    apply(logger,
                          macro_log,
                          [#{mfa => {supervisor,restart,2},
                             line => 736,
                             file => "supervisor.erl"},
                           error,
                           #{label => {supervisor,shutdown},
                             report =>
                                 [{supervisor,State#state.name},
                                  {errorContext,shutdown},
                                  {reason,reached_max_restart_intensity},
                                  {offender,extract_child(Child)}]},
                           #{domain => [otp,sasl],
                             report_cb => fun logger:format_otp_report/1,
                             logger_formatter =>
                                 #{title => "SUPERVISOR REPORT"},
                             error_logger =>
                                 #{tag => error_report,
                                   type => supervisor_report}}]);
                false ->
                    ok
            end,
            {shutdown,del_child(Child, NState)}
    end.

restart(simple_one_for_one, Child, State0) ->
    #child{pid = OldPid,mfargs = {M,F,A}} = Child,
    State1 =
        case OldPid of
            {restarting,_} ->
                NRes = State0#state.dynamic_restarts - 1,
                State0#state{dynamic_restarts = NRes};
            _ ->
                State0
        end,
    State2 = dyn_erase(OldPid, State1),
    case do_start_child_i(M, F, A) of
        {ok,Pid} ->
            NState = dyn_store(Pid, A, State2),
            {ok,NState};
        {ok,Pid,_Extra} ->
            NState = dyn_store(Pid, A, State2),
            {ok,NState};
        {error,Error} ->
            ROldPid = restarting(OldPid),
            NRestarts = State2#state.dynamic_restarts + 1,
            State3 = State2#state{dynamic_restarts = NRestarts},
            NState = dyn_store(ROldPid, A, State3),
            case logger:allow(error, supervisor) of
                true ->
                    apply(logger,
                          macro_log,
                          [#{mfa => {supervisor,restart,3},
                             line => 763,
                             file => "supervisor.erl"},
                           error,
                           #{label => {supervisor,start_error},
                             report =>
                                 [{supervisor,NState#state.name},
                                  {errorContext,start_error},
                                  {reason,Error},
                                  {offender,extract_child(Child)}]},
                           #{domain => [otp,sasl],
                             report_cb => fun logger:format_otp_report/1,
                             logger_formatter =>
                                 #{title => "SUPERVISOR REPORT"},
                             error_logger =>
                                 #{tag => error_report,
                                   type => supervisor_report}}]);
                false ->
                    ok
            end,
            {{try_again,ROldPid},NState}
    end;
restart(one_for_one, #child{id = Id} = Child, State) ->
    OldPid = Child#child.pid,
    case do_start_child(State#state.name, Child) of
        {ok,Pid} ->
            NState = set_pid(Pid, Id, State),
            {ok,NState};
        {ok,Pid,_Extra} ->
            NState = set_pid(Pid, Id, State),
            {ok,NState};
        {error,Reason} ->
            NState = set_pid(restarting(OldPid), Id, State),
            case logger:allow(error, supervisor) of
                true ->
                    apply(logger,
                          macro_log,
                          [#{mfa => {supervisor,restart,3},
                             line => 777,
                             file => "supervisor.erl"},
                           error,
                           #{label => {supervisor,start_error},
                             report =>
                                 [{supervisor,State#state.name},
                                  {errorContext,start_error},
                                  {reason,Reason},
                                  {offender,extract_child(Child)}]},
                           #{domain => [otp,sasl],
                             report_cb => fun logger:format_otp_report/1,
                             logger_formatter =>
                                 #{title => "SUPERVISOR REPORT"},
                             error_logger =>
                                 #{tag => error_report,
                                   type => supervisor_report}}]);
                false ->
                    ok
            end,
            {{try_again,Id},NState}
    end;
restart(rest_for_one,
        #child{id = Id} = Child,
        #state{name = SupName} = State) ->
    {ChAfter,ChBefore} = split_child(Id, State#state.children),
    {Return,ChAfter2} =
        restart_multiple_children(Child, ChAfter, SupName),
    {Return,State#state{children = append(ChAfter2, ChBefore)}};
restart(one_for_all, Child, #state{name = SupName} = State) ->
    Children1 = del_child(Child#child.id, State#state.children),
    {Return,NChildren} =
        restart_multiple_children(Child, Children1, SupName),
    {Return,State#state{children = NChildren}}.

restart_multiple_children(Child, Children, SupName) ->
    Children1 = terminate_children(Children, SupName),
    case start_children(Children1, SupName) of
        {ok,NChildren} ->
            {ok,NChildren};
        {error,NChildren,{failed_to_start_child,FailedId,_Reason}} ->
            NewPid =
                if
                    FailedId =:= Child#child.id ->
                        restarting(Child#child.pid);
                    true ->
                        {restarting,undefined}
                end,
            {{try_again,FailedId},set_pid(NewPid, FailedId, NChildren)}
    end.

restarting(Pid) when is_pid(Pid) ->
    {restarting,Pid};
restarting(RPid) ->
    RPid.

-spec try_again_restart(child_id() | {restarting, pid()}) -> ok.

try_again_restart(TryAgainId) ->
    gen_server:cast(self(), {try_again_restart,TryAgainId}).

terminate_children(Children, SupName) ->
    Terminate =
        fun(_Id, Child) when Child#child.restart_type =:= temporary ->
               do_terminate(Child, SupName),
               remove;
           (_Id, Child) ->
               do_terminate(Child, SupName),
               {update,Child#child{pid = undefined}}
        end,
    {ok,NChildren} = children_map(Terminate, Children),
    NChildren.

do_terminate(Child, SupName) when is_pid(Child#child.pid) ->
    case shutdown(Child#child.pid, Child#child.shutdown) of
        ok ->
            ok;
        {error,normal} when not (Child#child.restart_type =:= permanent) ->
            ok;
        {error,OtherReason} ->
            case logger:allow(error, supervisor) of
                true ->
                    apply(logger,
                          macro_log,
                          [#{mfa => {supervisor,do_terminate,2},
                             line => 838,
                             file => "supervisor.erl"},
                           error,
                           #{label => {supervisor,shutdown_error},
                             report =>
                                 [{supervisor,SupName},
                                  {errorContext,shutdown_error},
                                  {reason,OtherReason},
                                  {offender,extract_child(Child)}]},
                           #{domain => [otp,sasl],
                             report_cb => fun logger:format_otp_report/1,
                             logger_formatter =>
                                 #{title => "SUPERVISOR REPORT"},
                             error_logger =>
                                 #{tag => error_report,
                                   type => supervisor_report}}]);
                false ->
                    ok
            end
    end,
    ok;
do_terminate(_Child, _SupName) ->
    ok.

shutdown(Pid, brutal_kill) ->
    case monitor_child(Pid) of
        ok ->
            exit(Pid, kill),
            receive
                {'DOWN',_MRef,process,Pid,killed} ->
                    ok;
                {'DOWN',_MRef,process,Pid,OtherReason} ->
                    {error,OtherReason}
            end;
        {error,Reason} ->
            {error,Reason}
    end;
shutdown(Pid, Time) ->
    case monitor_child(Pid) of
        ok ->
            exit(Pid, shutdown),
            receive
                {'DOWN',_MRef,process,Pid,shutdown} ->
                    ok;
                {'DOWN',_MRef,process,Pid,OtherReason} ->
                    {error,OtherReason}
            after
                Time ->
                    exit(Pid, kill),
                    receive
                        {'DOWN',_MRef,process,Pid,OtherReason} ->
                            {error,OtherReason}
                    end
            end;
        {error,Reason} ->
            {error,Reason}
    end.

monitor_child(Pid) ->
    monitor(process, Pid),
    unlink(Pid),
    receive
        {'EXIT',Pid,Reason} ->
            receive
                {'DOWN',_,process,Pid,_} ->
                    {error,Reason}
            end
    after
        0 -> ok
    end.

terminate_dynamic_children(State) ->
    Child = get_dynamic_child(State),
    {Pids,EStack0} = monitor_dynamic_children(Child, State),
    Sz = sets:size(Pids),
    EStack =
        case Child#child.shutdown of
            brutal_kill ->
                sets:fold(fun(P, _) ->
                                 exit(P, kill)
                          end,
                          ok,
                          Pids),
                wait_dynamic_children(Child,
                                      Pids,
                                      Sz,
                                      undefined,
                                      EStack0);
            infinity ->
                sets:fold(fun(P, _) ->
                                 exit(P, shutdown)
                          end,
                          ok,
                          Pids),
                wait_dynamic_children(Child,
                                      Pids,
                                      Sz,
                                      undefined,
                                      EStack0);
            Time ->
                sets:fold(fun(P, _) ->
                                 exit(P, shutdown)
                          end,
                          ok,
                          Pids),
                TRef = erlang:start_timer(Time, self(), kill),
                wait_dynamic_children(Child, Pids, Sz, TRef, EStack0)
        end,
    dict:fold(fun(Reason, Ls, _) ->
                     case logger:allow(error, supervisor) of
                         true ->
                             apply(logger,
                                   macro_log,
                                   [#{mfa =>
                                          {supervisor,
                                           terminate_dynamic_children,
                                           1},
                                      line => 942,
                                      file => "supervisor.erl"},
                                    error,
                                    #{label =>
                                          {supervisor,shutdown_error},
                                      report =>
                                          [{supervisor,State#state.name},
                                           {errorContext,shutdown_error},
                                           {reason,Reason},
                                           {offender,
                                            extract_child(Child#child{pid =
                                                                          Ls})}]},
                                    #{domain => [otp,sasl],
                                      report_cb =>
                                          fun logger:format_otp_report/1,
                                      logger_formatter =>
                                          #{title => "SUPERVISOR REPORT"},
                                      error_logger =>
                                          #{tag => error_report,
                                            type => supervisor_report}}]);
                         false ->
                             ok
                     end
              end,
              ok,
              EStack).

monitor_dynamic_children(Child, State) ->
    dyn_fold(fun(P, {Pids,EStack}) when is_pid(P) ->
                    case monitor_child(P) of
                        ok ->
                            {sets:add_element(P, Pids),EStack};
                        {error,normal}
                            when
                                not (Child#child.restart_type
                                     =:=
                                     permanent) ->
                            {Pids,EStack};
                        {error,Reason} ->
                            {Pids,dict:append(Reason, P, EStack)}
                    end;
                ({restarting,_}, {Pids,EStack}) ->
                    {Pids,EStack}
             end,
             {sets:new(),dict:new()},
             State).

wait_dynamic_children(_Child, _Pids, 0, undefined, EStack) ->
    EStack;
wait_dynamic_children(_Child, _Pids, 0, TRef, EStack) ->
    _ = erlang:cancel_timer(TRef),
    receive
        {timeout,TRef,kill} ->
            EStack
    after
        0 -> EStack
    end;
wait_dynamic_children(#child{shutdown = brutal_kill} = Child,
                      Pids,
                      Sz,
                      TRef,
                      EStack) ->
    receive
        {'DOWN',_MRef,process,Pid,killed} ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  EStack);
        {'DOWN',_MRef,process,Pid,Reason} ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  dict:append(Reason, Pid, EStack))
    end;
wait_dynamic_children(Child, Pids, Sz, TRef, EStack) ->
    receive
        {'DOWN',_MRef,process,Pid,shutdown} ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  EStack);
        {'DOWN',_MRef,process,Pid,{shutdown,_}} ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  EStack);
        {'DOWN',_MRef,process,Pid,normal}
            when not (Child#child.restart_type =:= permanent) ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  EStack);
        {'DOWN',_MRef,process,Pid,Reason} ->
            wait_dynamic_children(Child,
                                  sets:del_element(Pid, Pids),
                                  Sz - 1,
                                  TRef,
                                  dict:append(Reason, Pid, EStack));
        {timeout,TRef,kill} ->
            sets:fold(fun(P, _) ->
                             exit(P, kill)
                      end,
                      ok,
                      Pids),
            wait_dynamic_children(Child, Pids, Sz, undefined, EStack)
    end.

-spec save_child(child_rec(), state()) -> state().

save_child(#child{mfargs = {M,F,_}} = Child, State)
    when Child#child.restart_type =:= temporary ->
    do_save_child(Child#child{mfargs = {M,F,undefined}}, State);
save_child(Child, State) ->
    do_save_child(Child, State).

-spec do_save_child(child_rec(), state()) -> state().

do_save_child(#child{id = Id} = Child,
              #state{children = {Ids,Db}} = State) ->
    State#state{children = {[Id|Ids],Db#{Id => Child}}}.

-spec del_child(child_rec(), state()) -> state();
               (child_id(), children()) -> children().

del_child(#child{pid = Pid}, State)
    when State#state.strategy =:= simple_one_for_one ->
    dyn_erase(Pid, State);
del_child(Child, State)
    when is_record(Child, child), is_record(State, state) ->
    NChildren = del_child(Child#child.id, State#state.children),
    State#state{children = NChildren};
del_child(Id, {Ids,Db}) ->
    case maps:get(Id, Db) of
        Child when Child#child.restart_type =:= temporary ->
            {lists:delete(Id, Ids),maps:remove(Id, Db)};
        Child ->
            {Ids,Db#{Id => Child#child{pid = undefined}}}
    end.

-spec split_child(child_id(), children()) -> {children(), children()}.

split_child(Id, {Ids,Db}) ->
    {IdsAfter,IdsBefore} = split_ids(Id, Ids, []),
    DbBefore = maps:with(IdsBefore, Db),
    #{Id := Ch} = DbAfter = maps:with(IdsAfter, Db),
    {{IdsAfter,DbAfter#{Id => Ch#child{pid = undefined}}},
     {IdsBefore,DbBefore}}.

split_ids(Id, [Id|Ids], After) ->
    {lists:reverse([Id|After]),Ids};
split_ids(Id, [Other|Ids], After) ->
    split_ids(Id, Ids, [Other|After]).

-spec find_child(pid() | child_id(), state()) ->
                    {ok, child_rec()} | error.

find_child(Pid, State)
    when is_pid(Pid), State#state.strategy =:= simple_one_for_one ->
    case find_dynamic_child(Pid, State) of
        error ->
            case find_dynamic_child(restarting(Pid), State) of
                error ->
                    case is_process_alive(Pid) of
                        true ->
                            error;
                        false ->
                            {ok,get_dynamic_child(State)}
                    end;
                Other ->
                    Other
            end;
        Other ->
            Other
    end;
find_child(Id, #state{children = {_Ids,Db}}) ->
    maps:find(Id, Db).

-spec find_child_and_args(IdOrPid, state()) -> {ok, child_rec()} | error
                             when
                                 IdOrPid ::
                                     pid() |
                                     {restarting, pid()} |
                                     child_id().

find_child_and_args(Pid, State)
    when State#state.strategy =:= simple_one_for_one ->
    case find_dynamic_child(Pid, State) of
        {ok,#child{mfargs = {M,F,_}} = Child} ->
            {ok,Args} = dyn_args(Pid, State),
            {ok,Child#child{mfargs = {M,F,Args}}};
        error ->
            error
    end;
find_child_and_args(Pid, State) when is_pid(Pid) ->
    find_child_by_pid(Pid, State);
find_child_and_args(Id, #state{children = {_Ids,Db}}) ->
    maps:find(Id, Db).

-spec find_dynamic_child(IdOrPid, state()) -> {ok, child_rec()} | error
                            when
                                IdOrPid ::
                                    pid() |
                                    {restarting, pid()} |
                                    child_id().

find_dynamic_child(Pid, State) ->
    case dyn_exists(Pid, State) of
        true ->
            Child = get_dynamic_child(State),
            {ok,Child#child{pid = Pid}};
        false ->
            error
    end.

-spec find_child_by_pid(IdOrPid, state()) -> {ok, child_rec()} | error
                           when IdOrPid :: pid() | {restarting, pid()}.

find_child_by_pid(Pid, #state{children = {_Ids,Db}}) ->
    Fun =
        fun(_Id, #child{pid = P} = Ch, _) when P =:= Pid ->
               throw(Ch);
           (_, _, error) ->
               error
        end,
    try
        maps:fold(Fun, error, Db)
    catch
        Child ->
            {ok,Child}
    end.

-spec get_dynamic_child(state()) -> child_rec().

get_dynamic_child(#state{children = {[Id],Db}}) ->
    #{Id := Child} = Db,
    Child.

-spec set_pid(term(), child_id(), state()) -> state();
             (term(), child_id(), children()) -> children().

set_pid(Pid, Id, #state{children = Children} = State) ->
    State#state{children = set_pid(Pid, Id, Children)};
set_pid(Pid, Id, {Ids,Db}) ->
    NewDb =
        maps:update_with(Id,
                         fun(Child) ->
                                Child#child{pid = Pid}
                         end,
                         Db),
    {Ids,NewDb}.

-spec remove_child(child_id(), state()) -> state().

remove_child(Id, #state{children = {Ids,Db}} = State) ->
    NewIds = lists:delete(Id, Ids),
    NewDb = maps:remove(Id, Db),
    State#state{children = {NewIds,NewDb}}.

-spec children_map(Fun, children()) ->
                      {ok, children()} | {error, children(), Reason}
                      when
                          Fun ::
                              fun((child_id(), child_rec()) ->
                                      {update, child_rec()} |
                                      remove |
                                      {abort, Reason}),
                          Reason :: term().

children_map(Fun, {Ids,Db}) ->
    children_map(Fun, Ids, Db, []).

children_map(Fun, [Id|Ids], Db, Acc) ->
    case Fun(Id, maps:get(Id, Db)) of
        {update,Child} ->
            children_map(Fun, Ids, Db#{Id => Child}, [Id|Acc]);
        remove ->
            children_map(Fun, Ids, maps:remove(Id, Db), Acc);
        {abort,Reason} ->
            {error,{lists:reverse(Ids) ++ [Id|Acc],Db},Reason}
    end;
children_map(_Fun, [], Db, Acc) ->
    {ok,{Acc,Db}}.

-spec children_to_list(Fun, children()) -> List
                          when
                              Fun ::
                                  fun((child_id(), child_rec()) -> Elem),
                              List :: [Elem],
                              Elem :: term().

children_to_list(Fun, {Ids,Db}) ->
    children_to_list(Fun, Ids, Db, []).

children_to_list(Fun, [Id|Ids], Db, Acc) ->
    children_to_list(Fun, Ids, Db, [Fun(Id, maps:get(Id, Db))|Acc]);
children_to_list(_Fun, [], _Db, Acc) ->
    lists:reverse(Acc).

-spec children_fold(Fun, Acc0, children()) -> Acc1
                       when
                           Fun ::
                               fun((child_id(), child_rec(), AccIn) ->
                                       AccOut),
                           Acc0 :: term(),
                           Acc1 :: term(),
                           AccIn :: term(),
                           AccOut :: term().

children_fold(Fun, Init, {_Ids,Db}) ->
    maps:fold(Fun, Init, Db).

-spec append(children(), children()) -> children().

append({Ids1,Db1}, {Ids2,Db2}) ->
    {Ids1 ++ Ids2,maps:merge(Db1, Db2)}.

init_state(SupName, Type, Mod, Args) ->
    set_flags(Type,
              #state{name = supname(SupName, Mod),
                     module = Mod,
                     args = Args}).

set_flags(Flags, State) ->
    try check_flags(Flags) of
        #{strategy := Strategy,
          intensity := MaxIntensity,
          period := Period} ->
            {ok,
             State#state{strategy = Strategy,
                         intensity = MaxIntensity,
                         period = Period}}
    catch
        Thrown ->
            Thrown
    end.

check_flags(SupFlags) when is_map(SupFlags) ->
% PATCH FOR maps:merge
    %do_check_flags(maps:merge(#{strategy => one_for_one,
    %                            intensity => 1,
    %                            period => 5},
    %                          SupFlags));
    do_check_flags(SupFlags);
check_flags({Strategy,MaxIntensity,Period}) ->
    check_flags(#{strategy => Strategy,
                  intensity => MaxIntensity,
                  period => Period});
check_flags(What) ->
    throw({invalid_type,What}).

do_check_flags(#{strategy := Strategy,
                 intensity := MaxIntensity,
                 period := Period} =
                   Flags) ->
    validStrategy(Strategy),
    validIntensity(MaxIntensity),
    validPeriod(Period),
    Flags.

validStrategy(simple_one_for_one) ->
    true;
validStrategy(one_for_one) ->
    true;
validStrategy(one_for_all) ->
    true;
validStrategy(rest_for_one) ->
    true;
validStrategy(What) ->
    throw({invalid_strategy,What}).

validIntensity(Max) when is_integer(Max), Max >= 0 ->
    true;
validIntensity(What) ->
    throw({invalid_intensity,What}).

validPeriod(Period) when is_integer(Period), Period > 0 ->
    true;
validPeriod(What) ->
    throw({invalid_period,What}).

supname(self, Mod) ->
    {self(),Mod};
supname(N, _) ->
    N.

check_startspec(Children) ->
    check_startspec(Children, [], #{}).

check_startspec([ChildSpec|T], Ids, Db) ->
    case check_childspec(ChildSpec) of
        {ok,#child{id = Id} = Child} ->
            case maps:is_key(Id, Db) of
                true ->
                    {duplicate_child_name,Id};
                false ->
                    check_startspec(T, [Id|Ids], Db#{Id => Child})
            end;
        Error ->
            Error
    end;
check_startspec([], Ids, Db) ->
    {ok,{lists:reverse(Ids),Db}}.

check_childspec(ChildSpec) when is_map(ChildSpec) ->
    % PATCH FOR maps:merge/2
    %catch
    %    do_check_childspec(maps:merge(#{restart => permanent,
    %                                    type => worker},
    %                                  ChildSpec));
    catch
        do_check_childspec(ChildSpec);
check_childspec({Id,Func,RestartType,Shutdown,ChildType,Mods}) ->
    check_childspec(#{id => Id,
                      start => Func,
                      restart => RestartType,
                      shutdown => Shutdown,
                      type => ChildType,
                      modules => Mods});
check_childspec(X) ->
    {invalid_child_spec,X}.

do_check_childspec(#{restart := RestartType,type := ChildType} =
                       ChildSpec) ->
    Id =
        case ChildSpec of
            #{id := I} ->
                I;
            _ ->
                throw(missing_id)
        end,
    Func =
        case ChildSpec of
            #{start := F} ->
                F;
            _ ->
                throw(missing_start)
        end,
    validId(Id),
    validFunc(Func),
    validRestartType(RestartType),
    validChildType(ChildType),
    Shutdown =
        case ChildSpec of
            #{shutdown := S} ->
                S;
            #{type := worker} ->
                5000;
            #{type := supervisor} ->
                infinity
        end,
    validShutdown(Shutdown),
    Mods =
        case ChildSpec of
            #{modules := Ms} ->
                Ms;
            _ ->
                {M,_,_} = Func,
                [M]
        end,
    validMods(Mods),
    {ok,
     #child{id = Id,
            mfargs = Func,
            restart_type = RestartType,
            shutdown = Shutdown,
            child_type = ChildType,
            modules = Mods}}.

validChildType(supervisor) ->
    true;
validChildType(worker) ->
    true;
validChildType(What) ->
    throw({invalid_child_type,What}).

validId(_Id) ->
    true.

validFunc({M,F,A}) when is_atom(M), is_atom(F), is_list(A) ->
    true;
validFunc(Func) ->
    throw({invalid_mfa,Func}).

validRestartType(permanent) ->
    true;
validRestartType(temporary) ->
    true;
validRestartType(transient) ->
    true;
validRestartType(RestartType) ->
    throw({invalid_restart_type,RestartType}).

validShutdown(Shutdown) when is_integer(Shutdown), Shutdown > 0 ->
    true;
validShutdown(infinity) ->
    true;
validShutdown(brutal_kill) ->
    true;
validShutdown(Shutdown) ->
    throw({invalid_shutdown,Shutdown}).

validMods(dynamic) ->
    true;
validMods(Mods) when is_list(Mods) ->
    lists:foreach(fun(Mod) ->
                         if
                             is_atom(Mod) ->
                                 ok;
                             true ->
                                 throw({invalid_module,Mod})
                         end
                  end,
                  Mods);
validMods(Mods) ->
    throw({invalid_modules,Mods}).

child_to_spec(#child{id = Id,
                     mfargs = Func,
                     restart_type = RestartType,
                     shutdown = Shutdown,
                     child_type = ChildType,
                     modules = Mods}) ->
    #{id => Id,
      start => Func,
      restart => RestartType,
      shutdown => Shutdown,
      type => ChildType,
      modules => Mods}.

add_restart(State) ->
    I = State#state.intensity,
    P = State#state.period,
    R = State#state.restarts,
    Now = erlang:monotonic_time(1),
    R1 = add_restart([Now|R], Now, P),
    State1 = State#state{restarts = R1},
    case length(R1) of
        CurI when CurI =< I ->
            {ok,State1};
        _ ->
            {terminate,State1}
    end.

add_restart([R|Restarts], Now, Period) ->
    case inPeriod(R, Now, Period) of
        true ->
            [R|add_restart(Restarts, Now, Period)];
        _ ->
            []
    end;
add_restart([], _, _) ->
    [].

inPeriod(Then, Now, Period) ->
    Now =< Then + Period.

extract_child(Child) when is_list(Child#child.pid) ->
    [{nb_children,length(Child#child.pid)},
     {id,Child#child.id},
     {mfargs,Child#child.mfargs},
     {restart_type,Child#child.restart_type},
     {shutdown,Child#child.shutdown},
     {child_type,Child#child.child_type}];
extract_child(Child) ->
    [{pid,Child#child.pid},
     {id,Child#child.id},
     {mfargs,Child#child.mfargs},
     {restart_type,Child#child.restart_type},
     {shutdown,Child#child.shutdown},
     {child_type,Child#child.child_type}].

report_progress(Child, SupName) ->
    case logger:allow(info, supervisor) of
        true ->
            apply(logger,
                  macro_log,
                  [#{mfa => {supervisor,report_progress,2},
                     line => 1419,
                     file => "supervisor.erl"},
                   info,
                   #{label => {supervisor,progress},
                     report =>
                         [{supervisor,SupName},
                          {started,extract_child(Child)}]},
                   #{domain => [otp,sasl],
                     report_cb => fun logger:format_otp_report/1,
                     logger_formatter => #{title => "PROGRESS REPORT"},
                     error_logger =>
                         #{tag => info_report,type => progress}}]);
        false ->
            ok
    end.

format_status(terminate, [_PDict,State]) ->
    State;
format_status(_, [_PDict,State]) ->
    [{data,[{"State",State}]},
     {supervisor,[{"Callback",State#state.module}]}].

dyn_size(#state{dynamics = {Mod,Db}}) ->
    Mod:size(Db).

dyn_erase(Pid, #state{dynamics = {sets,Db}} = State) ->
    State#state{dynamics = {sets,sets:del_element(Pid, Db)}};
dyn_erase(Pid, #state{dynamics = {maps,Db}} = State) ->
    State#state{dynamics = {maps,maps:remove(Pid, Db)}}.

dyn_store(Pid, _, #state{dynamics = {sets,Db}} = State) ->
    State#state{dynamics = {sets,sets:add_element(Pid, Db)}};
dyn_store(Pid, Args, #state{dynamics = {maps,Db}} = State) ->
    State#state{dynamics = {maps,Db#{Pid => Args}}}.

dyn_fold(Fun, Init, #state{dynamics = {sets,Db}}) ->
    sets:fold(Fun, Init, Db);
dyn_fold(Fun, Init, #state{dynamics = {maps,Db}}) ->
    maps:fold(fun(Pid, _, Acc) ->
                     Fun(Pid, Acc)
              end,
              Init,
              Db).

dyn_map(Fun, #state{dynamics = {sets,Db}}) ->
    lists:map(Fun, sets:to_list(Db));
dyn_map(Fun, #state{dynamics = {maps,Db}}) ->
    lists:map(Fun, maps:keys(Db)).

dyn_exists(Pid, #state{dynamics = {sets,Db}}) ->
    sets:is_element(Pid, Db);
dyn_exists(Pid, #state{dynamics = {maps,Db}}) ->
    maps:is_key(Pid, Db).

dyn_args(_Pid, #state{dynamics = {sets,_Db}}) ->
    {ok,undefined};
dyn_args(Pid, #state{dynamics = {maps,Db}}) ->
    maps:find(Pid, Db).

dyn_init(State) ->
    dyn_init(get_dynamic_child(State), State).

dyn_init(Child, State) when Child#child.restart_type =:= temporary ->
    State#state{dynamics = {sets,sets:new()}};
dyn_init(_Child, State) ->
    State#state{dynamics = {maps,maps:new()}}.



