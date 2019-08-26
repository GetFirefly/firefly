-file("/home/build/elixir/lib/elixir/lib/supervisor.ex", 1).

-module('Elixir.Supervisor').

-callback init(init_arg :: term()) ->
                  {ok,
                   {supervisor:sup_flags(), [supervisor:child_spec()]}} |
                  ignore.

-spec which_children(supervisor()) ->
                        [{term() | undefined,
                          child() | restarting,
                          worker | supervisor,
                          supervisor:modules()}].

-spec terminate_child(supervisor(), term()) -> ok | {error, error}
                         when error :: not_found | simple_one_for_one.

-spec stop(supervisor(), reason :: term(), timeout()) -> ok.

-spec start_link(module(), term(), 'Elixir.GenServer':options()) ->
                    on_start().

-spec start_link([supervisor:child_spec() |
                  {module(), term()} |
                  module()],
                 options()) ->
                    on_start();
                (module(), term()) -> on_start().

-spec start_child(supervisor(),
                  supervisor:child_spec() |
                  {module(), term()} |
                  module() |
                  [term()]) ->
                     on_start_child().

-spec restart_child(supervisor(), term()) ->
                       {ok, child()} |
                       {ok, child(), term()} |
                       {error, error}
                       when
                           error ::
                               not_found |
                               simple_one_for_one |
                               running |
                               restarting |
                               term().

-spec init([supervisor:child_spec() | {module(), term()} | module()],
           [init_option()]) ->
              {ok, tuple()}.

-spec delete_child(supervisor(), term()) -> ok | {error, error}
                      when
                          error ::
                              not_found |
                              simple_one_for_one |
                              running |
                              restarting.

-spec count_children(supervisor()) ->
                        #{specs := non_neg_integer(),
                          active := non_neg_integer(),
                          supervisors := non_neg_integer(),
                          workers := non_neg_integer()}.

-spec child_spec(child_spec() | {module(), arg :: term()} | module(),
                 elixir:keyword()) ->
                    child_spec().

-export_type([child_spec/0]).

-type child_spec() ::
          #{id := atom() | term(),
            start := {module(), atom(), [term()]},
            restart => permanent | transient | temporary,
            shutdown => timeout() | brutal_kill,
            type => worker | supervisor,
            modules => [module()] | dynamic}.

-export_type([strategy/0]).

-type strategy() :: one_for_one | one_for_all | rest_for_one.

-export_type([init_option/0]).

-type init_option() ::
          {strategy, strategy()} |
          {max_restarts, non_neg_integer()} |
          {max_seconds, pos_integer()}.

-export_type([supervisor/0]).

-type supervisor() :: pid() | name() | {atom(), node()}.

-export_type([options/0]).

-type options() :: [option(), ...].

-export_type([option/0]).

-type option() :: {name, name()} | init_option().

-export_type([name/0]).

-type name() :: atom() | {global, term()} | {via, module(), term()}.

-export_type([child/0]).

-type child() :: pid() | undefined.

-export_type([on_start_child/0]).

-type on_start_child() ::
          {ok, child()} |
          {ok, child(), info :: term()} |
          {error, {already_started, child()} | already_present | term()}.

-export_type([on_start/0]).

-type on_start() ::
          {ok, pid()} |
          ignore |
          {error,
           {already_started, pid()} | {shutdown, term()} | term()}.

-export(['MACRO-__using__'/2,
         '__info__'/1,
         child_spec/2,
         count_children/1,
         delete_child/2,
         init/2,
         restart_child/2,
         start_child/2,
         start_link/2,
         start_link/3,
         stop/1,
         stop/2,
         stop/3,
         terminate_child/2,
         which_children/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Supervisor';
'__info__'(functions) ->
    [{child_spec,2},
     {count_children,1},
     {delete_child,2},
     {init,2},
     {restart_child,2},
     {start_child,2},
     {start_link,2},
     {start_link,3},
     {stop,1},
     {stop,2},
     {stop,3},
     {terminate_child,2},
     {which_children,1}];
'__info__'(macros) ->
    [{'__using__',1}];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Supervisor', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Supervisor', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Supervisor', Key);
'__info__'(deprecated) ->
    [].

'MACRO-__using__'(_@CALLER, _opts@1) ->
    {'__block__',
     [],
     [{'=',[],[{opts,[],'Elixir.Supervisor'},_opts@1]},
      {'__block__',
       [{keep,{<<"lib/supervisor.ex">>,0}}],
       [{import,
         [{keep,{<<"lib/supervisor.ex">>,451}},
          {context,'Elixir.Supervisor'}],
         [{'__aliases__',
           [{keep,{<<"lib/supervisor.ex">>,451}},{alias,false}],
           ['Supervisor','Spec']}]},
        {'@',
         [{keep,{<<"lib/supervisor.ex">>,452}},
          {context,'Elixir.Supervisor'},
          {import,'Elixir.Kernel'}],
         [{behaviour,
           [{keep,{<<"lib/supervisor.ex">>,452}},
            {context,'Elixir.Supervisor'}],
           [{'__aliases__',
             [{keep,{<<"lib/supervisor.ex">>,452}},{alias,false}],
             ['Supervisor']}]}]},
        {'if',
         [{keep,{<<"lib/supervisor.ex">>,454}},
          {context,'Elixir.Supervisor'},
          {import,'Elixir.Kernel'}],
         [{'==',
           [{keep,{<<"lib/supervisor.ex">>,454}},
            {context,'Elixir.Supervisor'},
            {import,'Elixir.Kernel'}],
           [{{'.',
              [{keep,{<<"lib/supervisor.ex">>,454}}],
              [{'__aliases__',
                [{keep,{<<"lib/supervisor.ex">>,454}},{alias,false}],
                ['Module']},
               get_attribute]},
             [{keep,{<<"lib/supervisor.ex">>,454}}],
             [{'__MODULE__',
               [{keep,{<<"lib/supervisor.ex">>,454}}],
               'Elixir.Supervisor'},
              doc]},
            nil]},
          [{do,
            {'@',
             [{keep,{<<"lib/supervisor.ex">>,455}},
              {context,'Elixir.Supervisor'},
              {import,'Elixir.Kernel'}],
             [{doc,
               [{keep,{<<"lib/supervisor.ex">>,455}},
                {context,'Elixir.Supervisor'}],
               [<<"Returns a specification to start this module under a"
                  " supervisor.\n\nSee `Supervisor`.\n">>]}]}}]]},
        {def,
         [{keep,{<<"lib/supervisor.ex">>,462}},
          {context,'Elixir.Supervisor'},
          {import,'Elixir.Kernel'}],
         [{child_spec,
           [{keep,{<<"lib/supervisor.ex">>,462}},
            {context,'Elixir.Supervisor'}],
           [{init_arg,
             [{keep,{<<"lib/supervisor.ex">>,462}}],
             'Elixir.Supervisor'}]},
          [{do,
            {'__block__',
             [{keep,{<<"lib/supervisor.ex">>,0}}],
             [{'=',
               [{keep,{<<"lib/supervisor.ex">>,463}}],
               [{default,
                 [{keep,{<<"lib/supervisor.ex">>,463}}],
                 'Elixir.Supervisor'},
                {'%{}',
                 [{keep,{<<"lib/supervisor.ex">>,463}}],
                 [{id,
                   {'__MODULE__',
                    [{keep,{<<"lib/supervisor.ex">>,464}}],
                    'Elixir.Supervisor'}},
                  {start,
                   {'{}',
                    [{keep,{<<"lib/supervisor.ex">>,465}}],
                    [{'__MODULE__',
                      [{keep,{<<"lib/supervisor.ex">>,465}}],
                      'Elixir.Supervisor'},
                     start_link,
                     [{init_arg,
                       [{keep,{<<"lib/supervisor.ex">>,465}}],
                       'Elixir.Supervisor'}]]}},
                  {type,supervisor}]}]},
              {{'.',
                [{keep,{<<"lib/supervisor.ex">>,469}}],
                [{'__aliases__',
                  [{keep,{<<"lib/supervisor.ex">>,469}},{alias,false}],
                  ['Supervisor']},
                 child_spec]},
               [{keep,{<<"lib/supervisor.ex">>,469}}],
               [{default,
                 [{keep,{<<"lib/supervisor.ex">>,469}}],
                 'Elixir.Supervisor'},
                {unquote,
                 [{keep,{<<"lib/supervisor.ex">>,469}}],
                 [{{'.',
                    [{keep,{<<"lib/supervisor.ex">>,469}}],
                    [{'__aliases__',
                      [{keep,{<<"lib/supervisor.ex">>,469}},
                       {alias,false}],
                      ['Macro']},
                     escape]},
                   [{keep,{<<"lib/supervisor.ex">>,469}}],
                   [{opts,
                     [{keep,{<<"lib/supervisor.ex">>,469}}],
                     'Elixir.Supervisor'}]}]}]}]}}]]},
        {defoverridable,
         [{keep,{<<"lib/supervisor.ex">>,472}},
          {context,'Elixir.Supervisor'},
          {import,'Elixir.Kernel'}],
         [[{child_spec,1}]]}]}]}.

call(_supervisor@1, _req@1) ->
    'Elixir.GenServer':call(_supervisor@1, _req@1, infinity).

child_spec({_,_,_,_,_,_} = _tuple@1, __overrides@1) ->
    error('Elixir.ArgumentError':exception(<<"old tuple-based child spe"
                                             "cification ",
                                             ('Elixir.Kernel':inspect(_tuple@1))/binary,
                                             " ",
                                             "is not supported in Super"
                                             "visor.child_spec/2">>));
child_spec(_module_or_map@1, _overrides@1) ->
    'Elixir.Enum':reduce(_overrides@1,
                         init_child(_module_or_map@1),
                         fun({_key@1,_value@1}, _acc@1)
                                when
                                    _key@1 =:= start
                                    orelse
                                    _key@1 =:= restart
                                    orelse
                                    _key@1 =:= shutdown
                                    orelse
                                    _key@1 =:= type
                                    orelse
                                    _key@1 =:= modules
                                    orelse
                                    _key@1 =:= id ->
                                _acc@1#{_key@1 => _value@1};
                            ({_key@2,__value@1}, __acc@1) ->
                                error('Elixir.ArgumentError':exception(<<"unkno"
                                                                         "wn ke"
                                                                         "y ",
                                                                         ('Elixir.Kernel':inspect(_key@2))/binary,
                                                                         " in c"
                                                                         "hild "
                                                                         "speci"
                                                                         "ficat"
                                                                         "ion o"
                                                                         "verri"
                                                                         "de">>))
                         end).

child_spec_error(_module@1) ->
    case 'Elixir.Code':'ensure_loaded?'(_module@1) of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            <<"The module ",
              ('Elixir.Kernel':inspect(_module@1))/binary,
              " was given as a child to a supervisor but it does not ex"
              "ist.">>;
        _ ->
            <<"The module ",
              ('Elixir.Kernel':inspect(_module@1))/binary,
              " was given as a child to a supervisor\nbut it does not i"
              "mplement child_spec/1.\n\nIf you own the given module, p"
              "lease define a child_spec/1 function\nthat receives an a"
              "rgument and returns a child specification as a map.\nFor"
              " example:\n\n    def child_spec(opts) do\n      %{\n    "
              "    id: __MODULE__,\n        start: {__MODULE__, :start_"
              "link, [opts]},\n        type: :worker,\n        restart:"
              " :permanent,\n        shutdown: 500\n      }\n    end\n"
              "\nNote that \"use Agent\", \"use GenServer\" and so on a"
              "utomatically define\nthis function for you.\n\nHowever, "
              "if you don't own the given module and it doesn't impleme"
              "nt\nchild_spec/1, instead of passing the module name dir"
              "ectly as a supervisor\nchild, you will have to pass a ch"
              "ild specification as a map:\n\n    %{\n      id: ",
              ('Elixir.Kernel':inspect(_module@1))/binary,
              ",\n      start: {",
              ('Elixir.Kernel':inspect(_module@1))/binary,
              ", :start_link, [arg1, arg2]}\n    }\n\nSee the Superviso"
              "r documentation for more information.\n">>
    end.

count_children(_supervisor@1) ->
    maps:from_list(call(_supervisor@1, count_children)).

delete_child(_supervisor@1, _child_id@1) ->
    call(_supervisor@1, {delete_child,_child_id@1}).

init(_children@1, _options@1)
    when
        is_list(_children@1)
        andalso
        is_list(_options@1) ->
    _strategy@1 = 'Elixir.Access':get(_options@1, strategy, nil),
    case _strategy@1 of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            error('Elixir.ArgumentError':exception(<<"expected :strateg"
                                                     "y option to be gi"
                                                     "ven">>));
        _ ->
            nil
    end,
    _intensity@1 = 'Elixir.Keyword':get(_options@1, max_restarts, 3),
    _period@1 = 'Elixir.Keyword':get(_options@1, max_seconds, 5),
    _flags@1 =
        #{strategy => _strategy@1,
          intensity => _intensity@1,
          period => _period@1},
    {ok,{_flags@1,'Elixir.Enum':map(_children@1, fun init_child/1)}}.

init_child(_module@1) when is_atom(_module@1) ->
    init_child({_module@1,[]});
init_child({_module@1,_arg@1}) when is_atom(_module@1) ->
    try
        _module@1:child_spec(_arg@1)
    catch
        error:__@3:___STACKTRACE__@1 when __@3 == undef ->
            _e@1 =
                'Elixir.Exception':normalize(error,
                                             __@3,
                                             ___STACKTRACE__@1),
            case ___STACKTRACE__@1 of
                [{_module@1,child_spec,[_arg@1],_}|_] ->
                    error('Elixir.ArgumentError':exception(child_spec_error(_module@1)));
                _stack@1 ->
                    erlang:raise(error,
                                 'Elixir.Kernel.Utils':raise(_e@1),
                                 _stack@1)
            end;
        error:#{'__struct__' := __@4,'__exception__' := true} = __@3:___STACKTRACE__@1
            when __@4 == 'Elixir.UndefinedFunctionError' ->
            _e@1 =
                'Elixir.Exception':normalize(error,
                                             __@3,
                                             ___STACKTRACE__@1),
            case ___STACKTRACE__@1 of
                [{_module@1,child_spec,[_arg@1],_}|_] ->
                    error('Elixir.ArgumentError':exception(child_spec_error(_module@1)));
                _stack@1 ->
                    erlang:raise(error,
                                 'Elixir.Kernel.Utils':raise(_e@1),
                                 _stack@1)
            end
    end;
init_child(_map@1) when is_map(_map@1) ->
    _map@1;
init_child({_,_,_,_,_,_} = _tuple@1) ->
    _tuple@1;
init_child(_other@1) ->
    error('Elixir.ArgumentError':exception(<<"supervisors expect each c"
                                             "hild to be one of the fol"
                                             "lowing:\n\n  * a module\n"
                                             "  * a {module, arg} tuple"
                                             "\n  * a child specificati"
                                             "on as a map with at least"
                                             " the :id and :start field"
                                             "s\n  * or a tuple with 6 "
                                             "elements generated by Sup"
                                             "ervisor.Spec (deprecated)"
                                             "\n\nGot: ",
                                             ('Elixir.Kernel':inspect(_other@1))/binary,
                                             "\n">>)).

restart_child(_supervisor@1, _child_id@1) ->
    call(_supervisor@1, {restart_child,_child_id@1}).

start_child(_supervisor@1, {_,_,_,_,_,_} = _child_spec@1) ->
    call(_supervisor@1, {start_child,_child_spec@1});
start_child(_supervisor@1, _args@1) when is_list(_args@1) ->
    call(_supervisor@1, {start_child,_args@1});
start_child(_supervisor@1, _child_spec@1) ->
    call(_supervisor@1,
         {start_child,'Elixir.Supervisor':child_spec(_child_spec@1, [])}).

start_link(_children@1, _options@1) when is_list(_children@1) ->
    {_sup_opts@1,_start_opts@1} =
        'Elixir.Keyword':split(_options@1,
                               [strategy,max_seconds,max_restarts]),
    start_link('Elixir.Supervisor.Default',
               init(_children@1, _sup_opts@1),
               _start_opts@1);
start_link(__@1, __@2) ->
    start_link(__@1, __@2, []).

start_link(_module@1, _init_arg@1, _options@1) when is_list(_options@1) ->
    case 'Elixir.Keyword':get(_options@1, name) of
        nil ->
            supervisor:start_link(_module@1, _init_arg@1);
        _atom@1 when is_atom(_atom@1) ->
            supervisor:start_link({local,_atom@1},
                                  _module@1,
                                  _init_arg@1);
        {global,__term@1} = _tuple@1 ->
            supervisor:start_link(_tuple@1, _module@1, _init_arg@1);
        {via,_via_module@1,__term@2} = _tuple@2
            when is_atom(_via_module@1) ->
            supervisor:start_link(_tuple@2, _module@1, _init_arg@1);
        _other@1 ->
            error('Elixir.ArgumentError':exception(<<"expected :name op"
                                                     "tion to be one of"
                                                     " the following:\n"
                                                     "\n  * nil\n  * at"
                                                     "om\n  * {:global,"
                                                     " term}\n  * {:via"
                                                     ", module, term}\n"
                                                     "\nGot: ",
                                                     ('Elixir.Kernel':inspect(_other@1))/binary,
                                                     "\n">>))
    end.

stop(__@1) ->
    stop(__@1, normal, infinity).

stop(__@1, __@2) ->
    stop(__@1, __@2, infinity).

stop(_supervisor@1, _reason@1, _timeout@1) ->
    'Elixir.GenServer':stop(_supervisor@1, _reason@1, _timeout@1).

terminate_child(_supervisor@1, _pid@1) when is_pid(_pid@1) ->
    call(_supervisor@1, {terminate_child,_pid@1});
terminate_child(_supervisor@1, _child_id@1) ->
    call(_supervisor@1, {terminate_child,_child_id@1}).

which_children(_supervisor@1) ->
    call(_supervisor@1, which_children).

