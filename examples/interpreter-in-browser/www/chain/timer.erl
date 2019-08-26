-file("timer.erl", 1).

-module(timer).

-export([apply_after/4,
         send_after/3,
         send_after/2,
         exit_after/3,
         exit_after/2,
         kill_after/2,
         kill_after/1,
         apply_interval/4,
         send_interval/3,
         send_interval/2,
         cancel/1,
         sleep/1,
         tc/1,
         tc/2,
         tc/3,
         now_diff/2,
         seconds/1,
         minutes/1,
         hours/1,
         hms/3]).

-export([start_link/0,
         start/0,
         handle_call/3,
         handle_info/2,
         init/1,
         code_change/3,
         handle_cast/2,
         terminate/2]).

-export([get_status/0]).

-export_type([tref/0]).

-opaque tref() :: {integer(), reference()}.

-type time() :: non_neg_integer().

-spec apply_after(Time, Module, Function, Arguments) ->
                     {ok, TRef} | {error, Reason}
                     when
                         Time :: time(),
                         Module :: module(),
                         Function :: atom(),
                         Arguments :: [term()],
                         TRef :: tref(),
                         Reason :: term().

apply_after(Time, M, F, A) ->
    req(apply_after, {Time,{M,F,A}}).

-spec send_after(Time, Pid, Message) -> {ok, TRef} | {error, Reason}
                    when
                        Time :: time(),
                        Pid :: pid() | (RegName :: atom()),
                        Message :: term(),
                        TRef :: tref(),
                        Reason :: term().

send_after(Time, Pid, Message) ->
    req(apply_after, {Time,{timer,send,[Pid,Message]}}).

-spec send_after(Time, Message) -> {ok, TRef} | {error, Reason}
                    when
                        Time :: time(),
                        Message :: term(),
                        TRef :: tref(),
                        Reason :: term().

send_after(Time, Message) ->
    send_after(Time, self(), Message).

-spec exit_after(Time, Pid, Reason1) -> {ok, TRef} | {error, Reason2}
                    when
                        Time :: time(),
                        Pid :: pid() | (RegName :: atom()),
                        TRef :: tref(),
                        Reason1 :: term(),
                        Reason2 :: term().

exit_after(Time, Pid, Reason) ->
    req(apply_after, {Time,{erlang,exit,[Pid,Reason]}}).

-spec exit_after(Time, Reason1) -> {ok, TRef} | {error, Reason2}
                    when
                        Time :: time(),
                        TRef :: tref(),
                        Reason1 :: term(),
                        Reason2 :: term().

exit_after(Time, Reason) ->
    exit_after(Time, self(), Reason).

-spec kill_after(Time, Pid) -> {ok, TRef} | {error, Reason2}
                    when
                        Time :: time(),
                        Pid :: pid() | (RegName :: atom()),
                        TRef :: tref(),
                        Reason2 :: term().

kill_after(Time, Pid) ->
    exit_after(Time, Pid, kill).

-spec kill_after(Time) -> {ok, TRef} | {error, Reason2}
                    when
                        Time :: time(),
                        TRef :: tref(),
                        Reason2 :: term().

kill_after(Time) ->
    exit_after(Time, self(), kill).

-spec apply_interval(Time, Module, Function, Arguments) ->
                        {ok, TRef} | {error, Reason}
                        when
                            Time :: time(),
                            Module :: module(),
                            Function :: atom(),
                            Arguments :: [term()],
                            TRef :: tref(),
                            Reason :: term().

apply_interval(Time, M, F, A) ->
    req(apply_interval, {Time,self(),{M,F,A}}).

-spec send_interval(Time, Pid, Message) -> {ok, TRef} | {error, Reason}
                       when
                           Time :: time(),
                           Pid :: pid() | (RegName :: atom()),
                           Message :: term(),
                           TRef :: tref(),
                           Reason :: term().

send_interval(Time, Pid, Message) ->
    req(apply_interval, {Time,Pid,{timer,send,[Pid,Message]}}).

-spec send_interval(Time, Message) -> {ok, TRef} | {error, Reason}
                       when
                           Time :: time(),
                           Message :: term(),
                           TRef :: tref(),
                           Reason :: term().

send_interval(Time, Message) ->
    send_interval(Time, self(), Message).

-spec cancel(TRef) -> {ok, cancel} | {error, Reason}
                when TRef :: tref(), Reason :: term().

cancel(BRef) ->
    req(cancel, BRef).

-spec sleep(Time) -> ok when Time :: timeout().

sleep(T) ->
    receive after T -> ok end.

-spec tc(Fun) -> {Time, Value}
            when Fun :: function(), Time :: integer(), Value :: term().

tc(F) ->
    T1 = erlang:monotonic_time(),
    Val = F(),
    T2 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T2 - T1, native, microsecond),
    {Time,Val}.

-spec tc(Fun, Arguments) -> {Time, Value}
            when
                Fun :: function(),
                Arguments :: [term()],
                Time :: integer(),
                Value :: term().

tc(F, A) ->
    T1 = erlang:monotonic_time(),
    Val = apply(F, A),
    T2 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T2 - T1, native, microsecond),
    {Time,Val}.

-spec tc(Module, Function, Arguments) -> {Time, Value}
            when
                Module :: module(),
                Function :: atom(),
                Arguments :: [term()],
                Time :: integer(),
                Value :: term().

tc(M, F, A) ->
    T1 = erlang:monotonic_time(),
    Val = apply(M, F, A),
    T2 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T2 - T1, native, microsecond),
    {Time,Val}.

-spec now_diff(T2, T1) -> Tdiff
                  when
                      T1 :: erlang:timestamp(),
                      T2 :: erlang:timestamp(),
                      Tdiff :: integer().

now_diff({A2,B2,C2}, {A1,B1,C1}) ->
    ((A2 - A1) * 1000000 + B2 - B1) * 1000000 + C2 - C1.

-spec seconds(Seconds) -> MilliSeconds
                 when
                     Seconds :: non_neg_integer(),
                     MilliSeconds :: non_neg_integer().

seconds(Seconds) ->
    1000 * Seconds.

-spec minutes(Minutes) -> MilliSeconds
                 when
                     Minutes :: non_neg_integer(),
                     MilliSeconds :: non_neg_integer().

minutes(Minutes) ->
    1000 * 60 * Minutes.

-spec hours(Hours) -> MilliSeconds
               when
                   Hours :: non_neg_integer(),
                   MilliSeconds :: non_neg_integer().

hours(Hours) ->
    1000 * 60 * 60 * Hours.

-spec hms(Hours, Minutes, Seconds) -> MilliSeconds
             when
                 Hours :: non_neg_integer(),
                 Minutes :: non_neg_integer(),
                 Seconds :: non_neg_integer(),
                 MilliSeconds :: non_neg_integer().

hms(H, M, S) ->
    hours(H) + minutes(M) + seconds(S).

-spec start() -> ok.

start() ->
    ensure_started().

-spec start_link() -> {ok, pid()} | {error, term()}.

start_link() ->
    gen_server:start_link({local,timer_server}, timer, [], []).

-spec init([]) -> {ok, [], infinity}.

init([]) ->
    process_flag(trap_exit, true),
    timer_tab = ets:new(timer_tab, [named_table,ordered_set,protected]),
    timer_interval_tab =
        ets:new(timer_interval_tab, [named_table,protected]),
    {ok,[],infinity}.

-spec ensure_started() -> ok.

ensure_started() ->
    case whereis(timer_server) of
        undefined ->
            C = {timer_server,
                 {timer,start_link,[]},
                 permanent,
                 1000,
                 worker,
                 [timer]},
            _ = supervisor:start_child(kernel_safe_sup, C),
            ok;
        _ ->
            ok
    end.

req(Req, Arg) ->
    SysTime = system_time(),
    ensure_started(),
    gen_server:call(timer_server, {Req,Arg,SysTime}, infinity).

-type timers() :: term().

-spec handle_call(term(), term(), timers()) ->
                     {reply, term(), timers(), timeout()} |
                     {noreply, timers(), timeout()}.

handle_call({apply_after,{Time,Op},Started}, _From, _Ts)
    when is_integer(Time), Time >= 0 ->
    BRef = {Started + 1000 * Time,make_ref()},
    Timer = {BRef,timeout,Op},
    ets:insert(timer_tab, Timer),
    Timeout = timer_timeout(system_time()),
    {reply,{ok,BRef},[],Timeout};
handle_call({apply_interval,{Time,To,MFA},Started}, _From, _Ts)
    when is_integer(Time), Time >= 0 ->
    case get_pid(To) of
        Pid when is_pid(Pid) ->
            catch link(Pid),
            SysTime = system_time(),
            Ref = make_ref(),
            BRef1 = {interval,Ref},
            Interval = Time * 1000,
            BRef2 = {Started + Interval,Ref},
            Timer = {BRef2,{repeat,Interval,Pid},MFA},
            ets:insert(timer_interval_tab, {BRef1,BRef2,Pid}),
            ets:insert(timer_tab, Timer),
            Timeout = timer_timeout(SysTime),
            {reply,{ok,BRef1},[],Timeout};
        _ ->
            {reply,{error,badarg},[],next_timeout()}
    end;
handle_call({cancel,BRef = {_Time,Ref},_}, _From, Ts)
    when is_reference(Ref) ->
    delete_ref(BRef),
    {reply,{ok,cancel},Ts,next_timeout()};
handle_call({cancel,_BRef,_}, _From, Ts) ->
    {reply,{error,badarg},Ts,next_timeout()};
handle_call({apply_after,_,_}, _From, Ts) ->
    {reply,{error,badarg},Ts,next_timeout()};
handle_call({apply_interval,_,_}, _From, Ts) ->
    {reply,{error,badarg},Ts,next_timeout()};
handle_call(_Else, _From, Ts) ->
    {noreply,Ts,next_timeout()}.

-spec handle_info(term(), timers()) -> {noreply, timers(), timeout()}.

handle_info(timeout, Ts) ->
    Timeout = timer_timeout(system_time()),
    {noreply,Ts,Timeout};
handle_info({'EXIT',Pid,_Reason}, Ts) ->
    pid_delete(Pid),
    {noreply,Ts,next_timeout()};
handle_info(_OtherMsg, Ts) ->
    {noreply,Ts,next_timeout()}.

-spec handle_cast(term(), timers()) -> {noreply, timers(), timeout()}.

handle_cast(_Req, Ts) ->
    {noreply,Ts,next_timeout()}.

-spec terminate(term(), _State) -> ok.

terminate(_Reason, _State) ->
    ok.

-spec code_change(term(), State, term()) -> {ok, State}.

code_change(_OldVsn, State, _Extra) ->
    {ok,State}.

timer_timeout(SysTime) ->
    case ets:first(timer_tab) of
        '$end_of_table' ->
            infinity;
        {Time,_Ref} when Time > SysTime ->
            Timeout = (Time - SysTime + 999) div 1000,
            min(Timeout, 8388608);
        Key ->
            case ets:lookup(timer_tab, Key) of
                [{Key,timeout,MFA}] ->
                    ets:delete(timer_tab, Key),
                    do_apply(MFA),
                    timer_timeout(SysTime);
                [{{Time,Ref},Repeat = {repeat,Interv,To},MFA}] ->
                    ets:delete(timer_tab, Key),
                    NewTime = Time + Interv,
                    ets:insert(timer_interval_tab,
                               {{interval,Ref},{NewTime,Ref},To}),
                    do_apply(MFA),
                    ets:insert(timer_tab, {{NewTime,Ref},Repeat,MFA}),
                    timer_timeout(SysTime)
            end
    end.

delete_ref(BRef = {interval,_}) ->
    case ets:lookup(timer_interval_tab, BRef) of
        [{_,BRef2,_Pid}] ->
            ets:delete(timer_interval_tab, BRef),
            ets:delete(timer_tab, BRef2);
        _ ->
            ok
    end;
delete_ref(BRef) ->
    ets:delete(timer_tab, BRef).

-spec pid_delete(pid()) -> ok.

pid_delete(Pid) ->
    IntervalTimerList =
        ets:select(timer_interval_tab,
                   [{{'_','_','$1'},[{'==','$1',Pid}],['$_']}]),
    lists:foreach(fun({IntKey,TimerKey,_}) ->
                         ets:delete(timer_interval_tab, IntKey),
                         ets:delete(timer_tab, TimerKey)
                  end,
                  IntervalTimerList).

-spec next_timeout() -> timeout().

next_timeout() ->
    case ets:first(timer_tab) of
        '$end_of_table' ->
            infinity;
        {Time,_} ->
            min(positive((Time - system_time() + 999) div 1000),
                8388608)
    end.

do_apply({M,F,A}) ->
    case {M,F,A} of
        {timer,send,A} ->
            catch send(A);
        {erlang,exit,[Name,Reason]} ->
            catch exit(get_pid(Name), Reason);
        _ ->
            catch spawn(M, F, A)
    end.

positive(X) ->
    max(X, 0).

system_time() ->
    erlang:monotonic_time(1000000).

send([Pid,Msg]) ->
    Pid ! Msg.

get_pid(Name) when is_pid(Name) ->
    Name;
get_pid(undefined) ->
    undefined;
get_pid(Name) when is_atom(Name) ->
    get_pid(whereis(Name));
get_pid(_) ->
    undefined.

-spec get_status() ->
                    {{timer_tab, non_neg_integer()},
                     {timer_interval_tab, non_neg_integer()}}.

get_status() ->
    Info1 = ets:info(timer_tab),
    {size,TotalNumTimers} = lists:keyfind(size, 1, Info1),
    Info2 = ets:info(timer_interval_tab),
    {size,NumIntervalTimers} = lists:keyfind(size, 1, Info2),
    {{timer_tab,TotalNumTimers},{timer_interval_tab,NumIntervalTimers}}.



