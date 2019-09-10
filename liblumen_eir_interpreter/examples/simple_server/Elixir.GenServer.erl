-file("/home/build/elixir/lib/elixir/lib/gen_server.ex", 1).

-module('Elixir.GenServer').

-callback terminate(reason, state :: term()) -> term()
                       when
                           reason ::
                               normal | shutdown | {shutdown, term()}.

-optional_callbacks([terminate/2]).

-callback init(init_arg :: term()) ->
                  {ok, state} |
                  {ok,
                   state,
                   timeout() | hibernate | {continue, term()}} |
                  ignore |
                  {stop, reason :: any()}
                  when state :: any().

-callback handle_info(msg :: timeout | term(), state :: term()) ->
                         {noreply, new_state} |
                         {noreply,
                          new_state,
                          timeout() | hibernate | {continue, term()}} |
                         {stop, reason :: term(), new_state}
                         when new_state :: term().

-optional_callbacks([handle_info/2]).

-callback handle_continue(continue :: term(), state :: term()) ->
                             {noreply, new_state} |
                             {noreply,
                              new_state,
                              timeout() | hibernate | {continue, term()}} |
                             {stop, reason :: term(), new_state}
                             when new_state :: term().

-optional_callbacks([handle_continue/2]).

-callback handle_cast(request :: term(), state :: term()) ->
                         {noreply, new_state} |
                         {noreply,
                          new_state,
                          timeout() | hibernate | {continue, term()}} |
                         {stop, reason :: term(), new_state}
                         when new_state :: term().

-optional_callbacks([handle_cast/2]).

-callback handle_call(request :: term(), from(), state :: term()) ->
                         {reply, reply, new_state} |
                         {reply,
                          reply,
                          new_state,
                          timeout() | hibernate | {continue, term()}} |
                         {noreply, new_state} |
                         {noreply,
                          new_state,
                          timeout() | hibernate | {continue, term()}} |
                         {stop, reason, reply, new_state} |
                         {stop, reason, new_state}
                         when
                             reply :: term(),
                             new_state :: term(),
                             reason :: term().

-optional_callbacks([handle_call/3]).

-callback format_status(reason, pdict_and_state :: list()) -> term()
                           when reason :: normal | terminate.

-optional_callbacks([format_status/2]).

-callback code_change(old_vsn, state :: term(), extra :: term()) ->
                         {ok, new_state :: term()} |
                         {error, reason :: term()} |
                         {down, term()}
                         when old_vsn :: term().

-optional_callbacks([code_change/3]).

-spec whereis(server()) -> pid() | {atom(), node()} | nil.

-spec stop(server(), reason :: term(), timeout()) -> ok.

-spec start_link(module(), any(), options()) -> on_start().

-spec start(module(), any(), options()) -> on_start().

-spec reply(from(), term()) -> ok.

-spec multi_call([node()], name :: atom(), term(), timeout()) ->
                    {replies :: [{node(), term()}],
                     bad_nodes :: [node()]}.

-spec cast(server(), term()) -> ok.

-spec call(server(), term(), timeout()) -> term().

-spec abcast([node()], name :: atom(), term()) -> abcast.

-export_type([from/0]).

-type from() :: {pid(), tag :: term()}.

-export_type([server/0]).

-type server() :: pid() | name() | {atom(), node()}.

-export_type([debug/0]).

-type debug() ::
          [trace | log | statistics | {log_to_file, 'Elixir.Path':t()}].

-export_type([option/0]).

-type option() ::
          {debug, debug()} |
          {name, name()} |
          {timeout, timeout()} |
          {spawn_opt, 'Elixir.Process':spawn_opt()} |
          {hibernate_after, timeout()}.

-export_type([options/0]).

-type options() :: [option()].

-export_type([name/0]).

-type name() :: atom() | {global, term()} | {via, module(), term()}.

-export_type([on_start/0]).

-type on_start() ::
          {ok, pid()} |
          ignore |
          {error, {already_started, pid()} | term()}.

-export(['MACRO-__before_compile__'/2,
         'MACRO-__using__'/2,
         '__info__'/1,
         abcast/2,
         abcast/3,
         call/2,
         call/3,
         cast/2,
         multi_call/2,
         multi_call/3,
         multi_call/4,
         reply/2,
         start/2,
         start/3,
         start_link/2,
         start_link/3,
         stop/1,
         stop/2,
         stop/3,
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
    'Elixir.GenServer';
'__info__'(functions) ->
    [{abcast,2},
     {abcast,3},
     {call,2},
     {call,3},
     {cast,2},
     {multi_call,2},
     {multi_call,3},
     {multi_call,4},
     {reply,2},
     {start,2},
     {start,3},
     {start_link,2},
     {start_link,3},
     {stop,1},
     {stop,2},
     {stop,3},
     {whereis,1}];
'__info__'(macros) ->
    [{'__before_compile__',1},{'__using__',1}];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.GenServer', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.GenServer', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.GenServer', Key);
'__info__'(deprecated) ->
    [].

'MACRO-__before_compile__'(_@CALLER, _env@1) ->
    case
        'Elixir.Module':'defines?'(case _env@1 of
                                       #{module := __@1} ->
                                           __@1;
                                       __@1 when is_map(__@1) ->
                                           error({badkey,module,__@1});
                                       __@1 ->
                                           __@1:module()
                                   end,
                                   {init,1})
    of
        __@2
            when
                __@2 =:= nil
                orelse
                __@2 =:= false ->
            _message@1 =
                <<"function init/1 required by behaviour GenServer is n"
                  "ot implemented (in module ",
                  ('Elixir.Kernel':inspect(case _env@1 of
                                               #{module := __@3} ->
                                                   __@3;
                                               __@3 when is_map(__@3) ->
                                                   error({badkey,
                                                          module,
                                                          __@3});
                                               __@3 ->
                                                   __@3:module()
                                           end))/binary,
                  ").\n\nWe will inject a default implementation for no"
                  "w:\n\n    def init(init_arg) do\n      {:ok, init_ar"
                  "g}\n    end\n\nYou can copy the implementation above"
                  " or define your own that converts the arguments give"
                  "n to GenServer.start_link/3 to the server state.\n">>,
            elixir_errors:warn(case _env@1 of
                                   #{line := __@4} ->
                                       __@4;
                                   __@4 when is_map(__@4) ->
                                       error({badkey,line,__@4});
                                   __@4 ->
                                       __@4:line()
                               end,
                               case _env@1 of
                                   #{file := __@5} ->
                                       __@5;
                                   __@5 when is_map(__@5) ->
                                       error({badkey,file,__@5});
                                   __@5 ->
                                       __@5:file()
                               end,
                               _message@1),
            {'__block__',
             [],
             [{'@',
               [{context,'Elixir.GenServer'},{import,'Elixir.Kernel'}],
               [{doc,[{context,'Elixir.GenServer'}],[false]}]},
              {def,
               [{context,'Elixir.GenServer'},{import,'Elixir.Kernel'}],
               [{init,
                 [{context,'Elixir.GenServer'}],
                 [{init_arg,[],'Elixir.GenServer'}]},
                [{do,{ok,{init_arg,[],'Elixir.GenServer'}}}]]},
              {defoverridable,
               [{context,'Elixir.GenServer'},{import,'Elixir.Kernel'}],
               [[{init,1}]]}]};
        _ ->
            nil
    end.

'MACRO-__using__'(_@CALLER, _opts@1) ->
    {'__block__',
     [],
     [{'=',[],[{opts,[],'Elixir.GenServer'},_opts@1]},
      {'__block__',
       [{keep,{<<"lib/gen_server.ex">>,0}}],
       [{'@',
         [{keep,{<<"lib/gen_server.ex">>,719}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{behaviour,
           [{keep,{<<"lib/gen_server.ex">>,719}},
            {context,'Elixir.GenServer'}],
           [{'__aliases__',
             [{keep,{<<"lib/gen_server.ex">>,719}},{alias,false}],
             ['GenServer']}]}]},
        {'if',
         [{keep,{<<"lib/gen_server.ex">>,721}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{'==',
           [{keep,{<<"lib/gen_server.ex">>,721}},
            {context,'Elixir.GenServer'},
            {import,'Elixir.Kernel'}],
           [{{'.',
              [{keep,{<<"lib/gen_server.ex">>,721}}],
              [{'__aliases__',
                [{keep,{<<"lib/gen_server.ex">>,721}},{alias,false}],
                ['Module']},
               get_attribute]},
             [{keep,{<<"lib/gen_server.ex">>,721}}],
             [{'__MODULE__',
               [{keep,{<<"lib/gen_server.ex">>,721}}],
               'Elixir.GenServer'},
              doc]},
            nil]},
          [{do,
            {'@',
             [{keep,{<<"lib/gen_server.ex">>,722}},
              {context,'Elixir.GenServer'},
              {import,'Elixir.Kernel'}],
             [{doc,
               [{keep,{<<"lib/gen_server.ex">>,722}},
                {context,'Elixir.GenServer'}],
               [<<"Returns a specification to start this module under a"
                  " supervisor.\n\nSee `Supervisor`.\n">>]}]}}]]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,729}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{child_spec,
           [{keep,{<<"lib/gen_server.ex">>,729}},
            {context,'Elixir.GenServer'}],
           [{init_arg,
             [{keep,{<<"lib/gen_server.ex">>,729}}],
             'Elixir.GenServer'}]},
          [{do,
            {'__block__',
             [{keep,{<<"lib/gen_server.ex">>,0}}],
             [{'=',
               [{keep,{<<"lib/gen_server.ex">>,730}}],
               [{default,
                 [{keep,{<<"lib/gen_server.ex">>,730}}],
                 'Elixir.GenServer'},
                {'%{}',
                 [{keep,{<<"lib/gen_server.ex">>,730}}],
                 [{id,
                   {'__MODULE__',
                    [{keep,{<<"lib/gen_server.ex">>,731}}],
                    'Elixir.GenServer'}},
                  {start,
                   {'{}',
                    [{keep,{<<"lib/gen_server.ex">>,732}}],
                    [{'__MODULE__',
                      [{keep,{<<"lib/gen_server.ex">>,732}}],
                      'Elixir.GenServer'},
                     start_link,
                     [{init_arg,
                       [{keep,{<<"lib/gen_server.ex">>,732}}],
                       'Elixir.GenServer'}]]}}]}]},
              {{'.',
                [{keep,{<<"lib/gen_server.ex">>,735}}],
                [{'__aliases__',
                  [{keep,{<<"lib/gen_server.ex">>,735}},{alias,false}],
                  ['Supervisor']},
                 child_spec]},
               [{keep,{<<"lib/gen_server.ex">>,735}}],
               [{default,
                 [{keep,{<<"lib/gen_server.ex">>,735}}],
                 'Elixir.GenServer'},
                {unquote,
                 [{keep,{<<"lib/gen_server.ex">>,735}}],
                 [{{'.',
                    [{keep,{<<"lib/gen_server.ex">>,735}}],
                    [{'__aliases__',
                      [{keep,{<<"lib/gen_server.ex">>,735}},
                       {alias,false}],
                      ['Macro']},
                     escape]},
                   [{keep,{<<"lib/gen_server.ex">>,735}}],
                   [{opts,
                     [{keep,{<<"lib/gen_server.ex">>,735}}],
                     'Elixir.GenServer'}]}]}]}]}}]]},
        {defoverridable,
         [{keep,{<<"lib/gen_server.ex">>,738}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [[{child_spec,1}]]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,741}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{before_compile,
           [{keep,{<<"lib/gen_server.ex">>,741}},
            {context,'Elixir.GenServer'}],
           [{'__aliases__',
             [{keep,{<<"lib/gen_server.ex">>,741}},{alias,false}],
             ['GenServer']}]}]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,743}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{doc,
           [{keep,{<<"lib/gen_server.ex">>,743}},
            {context,'Elixir.GenServer'}],
           [false]}]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,744}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{handle_call,
           [{keep,{<<"lib/gen_server.ex">>,744}},
            {context,'Elixir.GenServer'}],
           [{msg,
             [{keep,{<<"lib/gen_server.ex">>,744}}],
             'Elixir.GenServer'},
            {'_from',
             [{keep,{<<"lib/gen_server.ex">>,744}}],
             'Elixir.GenServer'},
            {state,
             [{keep,{<<"lib/gen_server.ex">>,744}}],
             'Elixir.GenServer'}]},
          [{do,
            {'__block__',
             [{keep,{<<"lib/gen_server.ex">>,0}}],
             [{'=',
               [{keep,{<<"lib/gen_server.ex">>,745}}],
               [{proc,
                 [{keep,{<<"lib/gen_server.ex">>,745}}],
                 'Elixir.GenServer'},
                {'case',
                 [{keep,{<<"lib/gen_server.ex">>,746}}],
                 [{{'.',
                    [{keep,{<<"lib/gen_server.ex">>,746}}],
                    [{'__aliases__',
                      [{keep,{<<"lib/gen_server.ex">>,746}},
                       {alias,false}],
                      ['Process']},
                     info]},
                   [{keep,{<<"lib/gen_server.ex">>,746}}],
                   [{self,
                     [{keep,{<<"lib/gen_server.ex">>,746}},
                      {context,'Elixir.GenServer'},
                      {import,'Elixir.Kernel'}],
                     []},
                    registered_name]},
                  [{do,
                    [{'->',
                      [{keep,{<<"lib/gen_server.ex">>,747}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,747}}],
                          'Elixir.GenServer'},
                         []}],
                       {self,
                        [{keep,{<<"lib/gen_server.ex">>,747}},
                         {context,'Elixir.GenServer'},
                         {import,'Elixir.Kernel'}],
                        []}]},
                     {'->',
                      [{keep,{<<"lib/gen_server.ex">>,748}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,748}}],
                          'Elixir.GenServer'},
                         {name,
                          [{keep,{<<"lib/gen_server.ex">>,748}}],
                          'Elixir.GenServer'}}],
                       {name,
                        [{keep,{<<"lib/gen_server.ex">>,748}}],
                        'Elixir.GenServer'}]}]}]]}]},
              {'case',
               [{keep,{<<"lib/gen_server.ex">>,752}}],
               [{{'.',
                  [{keep,{<<"lib/gen_server.ex">>,752}}],
                  [erlang,phash2]},
                 [{keep,{<<"lib/gen_server.ex">>,752}}],
                 [1,1]},
                [{do,
                  [{'->',
                    [{keep,{<<"lib/gen_server.ex">>,753}}],
                    [[0],
                     {raise,
                      [{keep,{<<"lib/gen_server.ex">>,754}},
                       {context,'Elixir.GenServer'},
                       {import,'Elixir.Kernel'}],
                      [{'<<>>',
                        [{keep,{<<"lib/gen_server.ex">>,754}}],
                        [<<"attempted to call GenServer ">>,
                         {'::',
                          [{keep,{<<"lib/gen_server.ex">>,754}}],
                          [{{'.',
                             [{keep,{<<"lib/gen_server.ex">>,754}}],
                             ['Elixir.Kernel',to_string]},
                            [{keep,{<<"lib/gen_server.ex">>,754}}],
                            [{inspect,
                              [{keep,{<<"lib/gen_server.ex">>,754}},
                               {context,'Elixir.GenServer'},
                               {import,'Elixir.Kernel'}],
                              [{proc,
                                [{keep,{<<"lib/gen_server.ex">>,754}}],
                                'Elixir.GenServer'}]}]},
                           {binary,
                            [{keep,{<<"lib/gen_server.ex">>,754}}],
                            'Elixir.GenServer'}]},
                         <<" but no handle_call/3 clause was provided">>]}]}]},
                   {'->',
                    [{keep,{<<"lib/gen_server.ex">>,756}}],
                    [[1],
                     {'{}',
                      [{keep,{<<"lib/gen_server.ex">>,757}}],
                      [stop,
                       {bad_call,
                        {msg,
                         [{keep,{<<"lib/gen_server.ex">>,757}}],
                         'Elixir.GenServer'}},
                       {state,
                        [{keep,{<<"lib/gen_server.ex">>,757}}],
                        'Elixir.GenServer'}]}]}]}]]}]}}]]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,761}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{doc,
           [{keep,{<<"lib/gen_server.ex">>,761}},
            {context,'Elixir.GenServer'}],
           [false]}]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,762}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{handle_info,
           [{keep,{<<"lib/gen_server.ex">>,762}},
            {context,'Elixir.GenServer'}],
           [{msg,
             [{keep,{<<"lib/gen_server.ex">>,762}}],
             'Elixir.GenServer'},
            {state,
             [{keep,{<<"lib/gen_server.ex">>,762}}],
             'Elixir.GenServer'}]},
          [{do,
            {'__block__',
             [{keep,{<<"lib/gen_server.ex">>,0}}],
             [{'=',
               [{keep,{<<"lib/gen_server.ex">>,763}}],
               [{proc,
                 [{keep,{<<"lib/gen_server.ex">>,763}}],
                 'Elixir.GenServer'},
                {'case',
                 [{keep,{<<"lib/gen_server.ex">>,764}}],
                 [{{'.',
                    [{keep,{<<"lib/gen_server.ex">>,764}}],
                    [{'__aliases__',
                      [{keep,{<<"lib/gen_server.ex">>,764}},
                       {alias,false}],
                      ['Process']},
                     info]},
                   [{keep,{<<"lib/gen_server.ex">>,764}}],
                   [{self,
                     [{keep,{<<"lib/gen_server.ex">>,764}},
                      {context,'Elixir.GenServer'},
                      {import,'Elixir.Kernel'}],
                     []},
                    registered_name]},
                  [{do,
                    [{'->',
                      [{keep,{<<"lib/gen_server.ex">>,765}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,765}}],
                          'Elixir.GenServer'},
                         []}],
                       {self,
                        [{keep,{<<"lib/gen_server.ex">>,765}},
                         {context,'Elixir.GenServer'},
                         {import,'Elixir.Kernel'}],
                        []}]},
                     {'->',
                      [{keep,{<<"lib/gen_server.ex">>,766}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,766}}],
                          'Elixir.GenServer'},
                         {name,
                          [{keep,{<<"lib/gen_server.ex">>,766}}],
                          'Elixir.GenServer'}}],
                       {name,
                        [{keep,{<<"lib/gen_server.ex">>,766}}],
                        'Elixir.GenServer'}]}]}]]}]},
              {'=',
               [{keep,{<<"lib/gen_server.ex">>,769}}],
               [{pattern,
                 [{keep,{<<"lib/gen_server.ex">>,769}}],
                 'Elixir.GenServer'},
                [126,
                 112,
                 32,
                 126,
                 112,
                 32,
                 114,
                 101,
                 99,
                 101,
                 105,
                 118,
                 101,
                 100,
                 32,
                 117,
                 110,
                 101,
                 120,
                 112,
                 101,
                 99,
                 116,
                 101,
                 100,
                 32,
                 109,
                 101,
                 115,
                 115,
                 97,
                 103,
                 101,
                 32,
                 105,
                 110,
                 32,
                 104,
                 97,
                 110,
                 100,
                 108,
                 101,
                 95,
                 105,
                 110,
                 102,
                 111,
                 47,
                 50,
                 58,
                 32,
                 126,
                 112,
                 126,
                 110]]},
              {{'.',
                [{keep,{<<"lib/gen_server.ex">>,770}}],
                [error_logger,error_msg]},
               [{keep,{<<"lib/gen_server.ex">>,770}}],
               [{pattern,
                 [{keep,{<<"lib/gen_server.ex">>,770}}],
                 'Elixir.GenServer'},
                [{'__MODULE__',
                  [{keep,{<<"lib/gen_server.ex">>,770}}],
                  'Elixir.GenServer'},
                 {proc,
                  [{keep,{<<"lib/gen_server.ex">>,770}}],
                  'Elixir.GenServer'},
                 {msg,
                  [{keep,{<<"lib/gen_server.ex">>,770}}],
                  'Elixir.GenServer'}]]},
              {noreply,
               {state,
                [{keep,{<<"lib/gen_server.ex">>,771}}],
                'Elixir.GenServer'}}]}}]]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,774}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{doc,
           [{keep,{<<"lib/gen_server.ex">>,774}},
            {context,'Elixir.GenServer'}],
           [false]}]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,775}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{handle_cast,
           [{keep,{<<"lib/gen_server.ex">>,775}},
            {context,'Elixir.GenServer'}],
           [{msg,
             [{keep,{<<"lib/gen_server.ex">>,775}}],
             'Elixir.GenServer'},
            {state,
             [{keep,{<<"lib/gen_server.ex">>,775}}],
             'Elixir.GenServer'}]},
          [{do,
            {'__block__',
             [{keep,{<<"lib/gen_server.ex">>,0}}],
             [{'=',
               [{keep,{<<"lib/gen_server.ex">>,776}}],
               [{proc,
                 [{keep,{<<"lib/gen_server.ex">>,776}}],
                 'Elixir.GenServer'},
                {'case',
                 [{keep,{<<"lib/gen_server.ex">>,777}}],
                 [{{'.',
                    [{keep,{<<"lib/gen_server.ex">>,777}}],
                    [{'__aliases__',
                      [{keep,{<<"lib/gen_server.ex">>,777}},
                       {alias,false}],
                      ['Process']},
                     info]},
                   [{keep,{<<"lib/gen_server.ex">>,777}}],
                   [{self,
                     [{keep,{<<"lib/gen_server.ex">>,777}},
                      {context,'Elixir.GenServer'},
                      {import,'Elixir.Kernel'}],
                     []},
                    registered_name]},
                  [{do,
                    [{'->',
                      [{keep,{<<"lib/gen_server.ex">>,778}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,778}}],
                          'Elixir.GenServer'},
                         []}],
                       {self,
                        [{keep,{<<"lib/gen_server.ex">>,778}},
                         {context,'Elixir.GenServer'},
                         {import,'Elixir.Kernel'}],
                        []}]},
                     {'->',
                      [{keep,{<<"lib/gen_server.ex">>,779}}],
                      [[{{'_',
                          [{keep,{<<"lib/gen_server.ex">>,779}}],
                          'Elixir.GenServer'},
                         {name,
                          [{keep,{<<"lib/gen_server.ex">>,779}}],
                          'Elixir.GenServer'}}],
                       {name,
                        [{keep,{<<"lib/gen_server.ex">>,779}}],
                        'Elixir.GenServer'}]}]}]]}]},
              {'case',
               [{keep,{<<"lib/gen_server.ex">>,783}}],
               [{{'.',
                  [{keep,{<<"lib/gen_server.ex">>,783}}],
                  [erlang,phash2]},
                 [{keep,{<<"lib/gen_server.ex">>,783}}],
                 [1,1]},
                [{do,
                  [{'->',
                    [{keep,{<<"lib/gen_server.ex">>,784}}],
                    [[0],
                     {raise,
                      [{keep,{<<"lib/gen_server.ex">>,785}},
                       {context,'Elixir.GenServer'},
                       {import,'Elixir.Kernel'}],
                      [{'<<>>',
                        [{keep,{<<"lib/gen_server.ex">>,785}}],
                        [<<"attempted to cast GenServer ">>,
                         {'::',
                          [{keep,{<<"lib/gen_server.ex">>,785}}],
                          [{{'.',
                             [{keep,{<<"lib/gen_server.ex">>,785}}],
                             ['Elixir.Kernel',to_string]},
                            [{keep,{<<"lib/gen_server.ex">>,785}}],
                            [{inspect,
                              [{keep,{<<"lib/gen_server.ex">>,785}},
                               {context,'Elixir.GenServer'},
                               {import,'Elixir.Kernel'}],
                              [{proc,
                                [{keep,{<<"lib/gen_server.ex">>,785}}],
                                'Elixir.GenServer'}]}]},
                           {binary,
                            [{keep,{<<"lib/gen_server.ex">>,785}}],
                            'Elixir.GenServer'}]},
                         <<" but no handle_cast/2 clause was provided">>]}]}]},
                   {'->',
                    [{keep,{<<"lib/gen_server.ex">>,787}}],
                    [[1],
                     {'{}',
                      [{keep,{<<"lib/gen_server.ex">>,788}}],
                      [stop,
                       {bad_cast,
                        {msg,
                         [{keep,{<<"lib/gen_server.ex">>,788}}],
                         'Elixir.GenServer'}},
                       {state,
                        [{keep,{<<"lib/gen_server.ex">>,788}}],
                        'Elixir.GenServer'}]}]}]}]]}]}}]]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,792}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{doc,
           [{keep,{<<"lib/gen_server.ex">>,792}},
            {context,'Elixir.GenServer'}],
           [false]}]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,793}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{terminate,
           [{keep,{<<"lib/gen_server.ex">>,793}},
            {context,'Elixir.GenServer'}],
           [{'_reason',
             [{keep,{<<"lib/gen_server.ex">>,793}}],
             'Elixir.GenServer'},
            {'_state',
             [{keep,{<<"lib/gen_server.ex">>,793}}],
             'Elixir.GenServer'}]},
          [{do,ok}]]},
        {'@',
         [{keep,{<<"lib/gen_server.ex">>,797}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{doc,
           [{keep,{<<"lib/gen_server.ex">>,797}},
            {context,'Elixir.GenServer'}],
           [false]}]},
        {def,
         [{keep,{<<"lib/gen_server.ex">>,798}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [{code_change,
           [{keep,{<<"lib/gen_server.ex">>,798}},
            {context,'Elixir.GenServer'}],
           [{'_old',
             [{keep,{<<"lib/gen_server.ex">>,798}}],
             'Elixir.GenServer'},
            {state,
             [{keep,{<<"lib/gen_server.ex">>,798}}],
             'Elixir.GenServer'},
            {'_extra',
             [{keep,{<<"lib/gen_server.ex">>,798}}],
             'Elixir.GenServer'}]},
          [{do,
            {ok,
             {state,
              [{keep,{<<"lib/gen_server.ex">>,799}}],
              'Elixir.GenServer'}}}]]},
        {defoverridable,
         [{keep,{<<"lib/gen_server.ex">>,802}},
          {context,'Elixir.GenServer'},
          {import,'Elixir.Kernel'}],
         [[{code_change,3},
           {terminate,2},
           {handle_info,2},
           {handle_cast,2},
           {handle_call,3}]]}]}]}.

abcast(__@1, __@2) ->
    abcast([node()|nodes()], __@1, __@2).

abcast(_nodes@1, _name@1, _request@1)
    when
        is_list(_nodes@1)
        andalso
        is_atom(_name@1) ->
    _msg@1 = cast_msg(_request@1),
    'Elixir.Enum':reduce(_nodes@1,
                         [],
                         fun(_node@1, __@1) ->
                                begin
                                    do_send({_name@1,_node@1}, _msg@1),
                                    nil
                                end
                         end),
    abcast.

call(__@1, __@2) ->
    call(__@1, __@2, 5000).

call(_server@1, _request@1, _timeout@1) ->
    case whereis(_server@1) of
        nil ->
            exit({noproc,
                  {'Elixir.GenServer',
                   call,
                   [_server@1,_request@1,_timeout@1]}});
        _pid@1 when _pid@1 == self() ->
            exit({calling_self,
                  {'Elixir.GenServer',
                   call,
                   [_server@1,_request@1,_timeout@1]}});
        _pid@2 ->
            try gen:call(_pid@2, '$gen_call', _request@1, _timeout@1) of
                {ok,_res@1} ->
                    _res@1
            catch
                exit:_reason@1 ->
                    exit({_reason@1,
                          {'Elixir.GenServer',
                           call,
                           [_server@1,_request@1,_timeout@1]}})
            end
    end.

cast({global,_name@1}, _request@1) ->
    try
        global:send(_name@1, cast_msg(_request@1)),
        ok
    catch
        _:_ ->
            ok
    end;
cast({via,_mod@1,_name@1}, _request@1) ->
    try
        _mod@1:send(_name@1, cast_msg(_request@1)),
        ok
    catch
        _:_ ->
            ok
    end;
cast({_name@1,_node@1}, _request@1)
    when
        is_atom(_name@1)
        andalso
        is_atom(_node@1) ->
    do_send({_name@1,_node@1}, cast_msg(_request@1));
cast(_dest@1, _request@1)
    when
        is_atom(_dest@1)
        orelse
        is_pid(_dest@1) ->
    do_send(_dest@1, cast_msg(_request@1)).

cast_msg(_req@1) ->
    {'$gen_cast',_req@1}.

do_send(_dest@1, _msg@1) ->
    try
        erlang:send(_dest@1, _msg@1),
        ok
    catch
        _:_ ->
            ok
    end.

do_start(_link@1, _module@1, _init_arg@1, _options@1) ->
    case 'Elixir.Keyword':pop(_options@1, name) of
        {nil,_opts@1} ->
            gen:start(gen_server,
                      _link@1,
                      _module@1,
                      _init_arg@1,
                      _opts@1);
        {_atom@1,_opts@2} when is_atom(_atom@1) ->
            gen:start(gen_server,
                      _link@1,
                      {local,_atom@1},
                      _module@1,
                      _init_arg@1,
                      _opts@2);
        {{global,__term@1} = _tuple@1,_opts@3} ->
            gen:start(gen_server,
                      _link@1,
                      _tuple@1,
                      _module@1,
                      _init_arg@1,
                      _opts@3);
        {{via,_via_module@1,__term@2} = _tuple@2,_opts@4}
            when is_atom(_via_module@1) ->
            gen:start(gen_server,
                      _link@1,
                      _tuple@2,
                      _module@1,
                      _init_arg@1,
                      _opts@4);
        {_other@1,_} ->
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

multi_call(__@1, __@2) ->
    multi_call([node()|nodes()], __@1, __@2, infinity).

multi_call(__@1, __@2, __@3) ->
    multi_call(__@1, __@2, __@3, infinity).

multi_call(_nodes@1, _name@1, _request@1, _timeout@1) ->
    gen_server:multi_call(_nodes@1, _name@1, _request@1, _timeout@1).

reply({_to@1,_tag@1}, _reply@1) when is_pid(_to@1) ->
    erlang:send(_to@1, {_tag@1,_reply@1}),
    ok.

start(__@1, __@2) ->
    start(__@1, __@2, []).

start(_module@1, _init_arg@1, _options@1)
    when
        is_atom(_module@1)
        andalso
        is_list(_options@1) ->
    do_start(nolink, _module@1, _init_arg@1, _options@1).

start_link(__@1, __@2) ->
    start_link(__@1, __@2, []).

start_link(_module@1, _init_arg@1, _options@1)
    when
        is_atom(_module@1)
        andalso
        is_list(_options@1) ->
    do_start(link, _module@1, _init_arg@1, _options@1).

stop(__@1) ->
    stop(__@1, normal, infinity).

stop(__@1, __@2) ->
    stop(__@1, __@2, infinity).

stop(_server@1, _reason@1, _timeout@1) ->
    case whereis(_server@1) of
        nil ->
            exit({noproc,
                  {'Elixir.GenServer',
                   stop,
                   [_server@1,_reason@1,_timeout@1]}});
        _pid@1 when _pid@1 == self() ->
            exit({calling_self,
                  {'Elixir.GenServer',
                   stop,
                   [_server@1,_reason@1,_timeout@1]}});
        _pid@2 ->
            try
                proc_lib:stop(_pid@2, _reason@1, _timeout@1)
            catch
                exit:_err@1 ->
                    exit({_err@1,
                          {'Elixir.GenServer',
                           stop,
                           [_server@1,_reason@1,_timeout@1]}})
            end
    end.

whereis(_pid@1) when is_pid(_pid@1) ->
    _pid@1;
whereis(_name@1) when is_atom(_name@1) ->
    'Elixir.Process':whereis(_name@1);
whereis({global,_name@1}) ->
    case global:whereis_name(_name@1) of
        _pid@1 when is_pid(_pid@1) ->
            _pid@1;
        undefined ->
            nil
    end;
whereis({via,_mod@1,_name@1}) ->
    case apply(_mod@1, whereis_name, [_name@1]) of
        _pid@1 when is_pid(_pid@1) ->
            _pid@1;
        undefined ->
            nil
    end;
whereis({_name@1,_local@1})
    when
        is_atom(_name@1)
        andalso
        _local@1 == node() ->
    'Elixir.Process':whereis(_name@1);
whereis({_name@1,_node@1} = _server@1)
    when
        is_atom(_name@1)
        andalso
        is_atom(_node@1) ->
    _server@1.

