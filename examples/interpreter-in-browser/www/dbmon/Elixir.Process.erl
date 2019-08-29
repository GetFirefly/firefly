-file("/home/build/elixir/lib/elixir/lib/process.ex", 1).

-module('Elixir.Process').

-spec whereis(atom()) -> pid() | port() | nil.

-spec unregister(atom()) -> true.

-spec unlink(pid() | port()) -> true.

-spec spawn(module(), atom(), list(), spawn_opts()) ->
               pid() | {pid(), reference()}.

-spec spawn(fun(() -> any()), spawn_opts()) ->
               pid() | {pid(), reference()}.

-spec sleep(timeout()) -> ok.

-spec send_after(pid() | atom(), term(), non_neg_integer(), [option]) ->
                    reference()
                    when option :: {abs, boolean()}.

-spec send(dest, msg, [option]) -> ok | noconnect | nosuspend
              when
                  dest :: dest(),
                  msg :: any(),
                  option :: noconnect | nosuspend.

-spec registered() -> [atom()].

-spec register(pid() | port(), atom()) -> true.

-spec read_timer(reference()) -> non_neg_integer() | false.

-spec put(term(), term()) -> term() | nil.

-spec monitor(pid() | {name, node()} | name) -> reference()
                 when name :: atom().

-spec list() -> [pid()].

-spec link(pid() | port()) -> true.

-spec info(pid(), atom() | [atom()]) ->
              {atom(), term()} | [{atom(), term()}] | nil.

-spec info(pid()) -> elixir:keyword() | nil.

-spec hibernate(module(), atom(), list()) -> no_return().

-spec group_leader(pid(), leader :: pid()) -> true.

-spec group_leader() -> pid().

-spec get_keys(term()) -> [term()].

-spec get_keys() -> [term()].

-spec get(term(), default :: term()) -> term().

-spec get() -> [{term(), term()}].

-spec flag(pid(), save_calls, 0..10000) -> 0..10000.

-spec flag(error_handler, module()) -> module();
          (max_heap_size, heap_size()) -> heap_size();
          (message_queue_data, erlang:message_queue_data()) ->
              erlang:message_queue_data();
          (min_bin_vheap_size, non_neg_integer()) -> non_neg_integer();
          (min_heap_size, non_neg_integer()) -> non_neg_integer();
          (monitor_nodes, term()) -> term();
          ({monitor_nodes, term()}, term()) -> term();
          (priority, priority_level()) -> priority_level();
          (save_calls, 0..10000) -> 0..10000;
          (sensitive, boolean()) -> boolean();
          (trap_exit, boolean()) -> boolean().

-spec exit(pid(), term()) -> true.

-spec demonitor(reference(), options :: [flush | info]) -> boolean().

-spec delete(term()) -> term() | nil.

-spec cancel_timer(reference(), options) ->
                      non_neg_integer() | false | ok
                      when
                          options ::
                              [{async, boolean()} | {info, boolean()}].

-spec 'alive?'(pid()) -> boolean().

-type priority_level() :: low | normal | high | max.

-type heap_size() ::
          non_neg_integer() |
          #{size := non_neg_integer(),
            kill := boolean(),
            error_logger := boolean()}.

-export_type([spawn_opts/0]).

-type spawn_opts() :: [spawn_opt()].

-export_type([spawn_opt/0]).

-type spawn_opt() ::
          link |
          monitor |
          {priority, low | normal | high} |
          {fullsweep_after, non_neg_integer()} |
          {min_heap_size, non_neg_integer()} |
          {min_bin_vheap_size, non_neg_integer()}.

-export_type([dest/0]).

-type dest() ::
          pid() |
          port() |
          (registered_name :: atom()) |
          {registered_name :: atom(), node()}.

-export(['__info__'/1,
         'alive?'/1,
         cancel_timer/1,
         cancel_timer/2,
         delete/1,
         demonitor/1,
         demonitor/2,
         exit/2,
         flag/2,
         flag/3,
         get/0,
         get/1,
         get/2,
         get_keys/0,
         get_keys/1,
         group_leader/0,
         group_leader/2,
         hibernate/3,
         info/1,
         info/2,
         link/1,
         list/0,
         monitor/1,
         put/2,
         read_timer/1,
         register/2,
         registered/0,
         send/3,
         send_after/3,
         send_after/4,
         sleep/1,
         spawn/2,
         spawn/4,
         unlink/1,
         unregister/1,
         whereis/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Process';
'__info__'(functions) ->
    [{'alive?',1},
     {cancel_timer,1},
     {cancel_timer,2},
     {delete,1},
     {demonitor,1},
     {demonitor,2},
     {exit,2},
     {flag,2},
     {flag,3},
     {get,0},
     {get,1},
     {get,2},
     {get_keys,0},
     {get_keys,1},
     {group_leader,0},
     {group_leader,2},
     {hibernate,3},
     {info,1},
     {info,2},
     {link,1},
     {list,0},
     {monitor,1},
     {put,2},
     {read_timer,1},
     {register,2},
     {registered,0},
     {send,3},
     {send_after,3},
     {send_after,4},
     {sleep,1},
     {spawn,2},
     {spawn,4},
     {unlink,1},
     {unregister,1},
     {whereis,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Process', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Process', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Process', Key);
'__info__'(deprecated) ->
    [].

'alive?'(_pid@1) ->
    is_process_alive(_pid@1).

cancel_timer(__@1) ->
    cancel_timer(__@1, []).

cancel_timer(_timer_ref@1, _options@1) ->
    erlang:cancel_timer(_timer_ref@1, _options@1).

delete(_key@1) ->
    nillify(erase(_key@1)).

demonitor(__@1) ->
    demonitor(__@1, []).

demonitor(_monitor_ref@1, _options@1) ->
    demonitor(_monitor_ref@1, _options@1).

exit(_pid@1, _reason@1) ->
    exit(_pid@1, _reason@1).

flag(_flag@1, _value@1) ->
    process_flag(_flag@1, _value@1).

flag(_pid@1, _flag@1, _value@1) ->
    process_flag(_pid@1, _flag@1, _value@1).

get() ->
    get().

get(__@1) ->
    get(__@1, nil).

get(_key@1, _default@1) ->
    case get(_key@1) of
        undefined ->
            _default@1;
        _value@1 ->
            _value@1
    end.

get_keys() ->
    get_keys().

get_keys(_value@1) ->
    get_keys(_value@1).

group_leader() ->
    group_leader().

group_leader(_pid@1, _leader@1) ->
    group_leader(_leader@1, _pid@1).

hibernate(_mod@1, _fun_name@1, _args@1) ->
    erlang:hibernate(_mod@1, _fun_name@1, _args@1).

info(_pid@1) ->
    nillify(process_info(_pid@1)).

info(_pid@1, registered_name) ->
    case process_info(_pid@1, registered_name) of
        undefined ->
            nil;
        [] ->
            {registered_name,[]};
        _other@1 ->
            _other@1
    end;
info(_pid@1, _spec@1)
    when
        is_atom(_spec@1)
        orelse
        is_list(_spec@1) ->
    nillify(process_info(_pid@1, _spec@1)).

link(_pid_or_port@1) ->
    link(_pid_or_port@1).

list() ->
    processes().

monitor(_item@1) ->
    monitor(process, _item@1).

nillify(undefined) ->
    nil;
nillify(_other@1) ->
    _other@1.

put(_key@1, _value@1) ->
    nillify(put(_key@1, _value@1)).

read_timer(_timer_ref@1) ->
    erlang:read_timer(_timer_ref@1).

register(_pid_or_port@1, _name@1)
    when
        is_atom(_name@1)
        andalso
        not (_name@1 =:= false
             orelse
             _name@1 =:= true
             orelse
             _name@1 =:= undefined
             orelse
             _name@1 =:= nil) ->
    try
        register(_name@1, _pid_or_port@1)
    catch
        error:badarg when node(_pid_or_port@1) /= node() ->
            _message@1 =
                <<"could not register ",
                  ('Elixir.Kernel':inspect(_pid_or_port@1))/binary,
                  " because it belongs to another node">>,
            error('Elixir.ArgumentError':exception(_message@1),
                  [_pid_or_port@1,_name@1]);
        error:badarg ->
            _message@2 =
                <<"could not register ",
                  ('Elixir.Kernel':inspect(_pid_or_port@1))/binary,
                  " with ",
                  "name ",
                  ('Elixir.Kernel':inspect(_name@1))/binary,
                  " because it is not alive, the name is already ",
                  "taken, or it has already been given another name">>,
            error('Elixir.ArgumentError':exception(_message@2),
                  [_pid_or_port@1,_name@1])
    end.

registered() ->
    registered().

send(_dest@1, _msg@1, _options@1) ->
    erlang:send(_dest@1, _msg@1, _options@1).

send_after(__@1, __@2, __@3) ->
    send_after(__@1, __@2, __@3, []).

send_after(_dest@1, _msg@1, _time@1, _opts@1) ->
    erlang:send_after(_time@1, _dest@1, _msg@1, _opts@1).

sleep(_timeout@1)
    when
        is_integer(_timeout@1)
        andalso
        _timeout@1 >= 0;
        _timeout@1 == infinity ->
    receive after _timeout@1 -> ok end.

spawn(_fun@1, _opts@1) ->
    spawn_opt(_fun@1, _opts@1).

spawn(_mod@1, _fun@1, _args@1, _opts@1) ->
    spawn_opt(_mod@1, _fun@1, _args@1, _opts@1).

unlink(_pid_or_port@1) ->
    unlink(_pid_or_port@1).

unregister(_name@1) ->
    unregister(_name@1).

whereis(_name@1) ->
    nillify(whereis(_name@1)).

