-file("lib/db_mon_demo/app_supervisor.ex", 1).

-module('Elixir.DbMonDemo.AppSupervisor').

-compile([no_auto_import]).

-spec init(any()) ->
              {ok,
               {#{intensity := any(),
                  period := any(),
                  strategy := any()},
                [any()]}}.

-behaviour('Elixir.Supervisor').

-export(['__info__'/1,child_spec/1,init/1,start_link/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.DbMonDemo.AppSupervisor';
'__info__'(functions) ->
    [{child_spec,1},{init,1},{start_link,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.DbMonDemo.AppSupervisor', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.DbMonDemo.AppSupervisor', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.DbMonDemo.AppSupervisor', Key);
'__info__'(deprecated) ->
    [].

init(_ast@1) ->
    _children@1 = [{'Elixir.DbMonDemo.WindowSupervisor',_ast@1}],
    'Elixir.Supervisor':init(_children@1, [{strategy,one_for_one}]).

start_link(_ast@1) ->
    'Elixir.Supervisor':start_link('Elixir.DbMonDemo.AppSupervisor',
                                   _ast@1,
                                   [{name,
                                     'Elixir.DbMonDemo.AppSupervisor'}]).

-file("lib/supervisor.ex", 462).

child_spec(__@1) ->
    __@2 =
        #{id => 'Elixir.DbMonDemo.AppSupervisor',
          start => {'Elixir.DbMonDemo.AppSupervisor',start_link,[__@1]},
          type => supervisor},
    'Elixir.Supervisor':child_spec(__@2, []).

