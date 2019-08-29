-file("lib/db_mon_demo/document_supervisor.ex", 1).

-module('Elixir.DbMonDemo.DocumentSupervisor').

-export(['__info__'/1,
         child_spec/1,
         handle_call/3,
         init/1,
         init_it/3,
         start_link/1,
         terminate/2]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.DbMonDemo.DocumentSupervisor';
'__info__'(functions) ->
    [{child_spec,1},
     {handle_call,3},
     {init,1},
     {init_it,3},
     {start_link,1},
     {terminate,2}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.DbMonDemo.DocumentSupervisor', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.DbMonDemo.DocumentSupervisor', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.DbMonDemo.DocumentSupervisor', Key);
'__info__'(deprecated) ->
    [].

child_spec(__@1) ->
    #{id => 'Elixir.DbMonDemo.DocumentSupervisor',
      start => {'Elixir.DbMonDemo.DocumentSupervisor',start_link,[__@1]},
      type => supervisor,
      restart => temporary,
      shutdown => 500}.

fetch_child_from_pid(__@1, __@2) ->
    case
        'Elixir.Enum':reduce_while(__@1,
                                   {error,0},
                                   fun({__@4,__@5,__@6} = __@3,
                                       {error,__@7})
                                          when __@2 =:= __@4 ->
                                          {halt,{__@3,__@7}};
                                      ({__@8,__@9,__@10}, {error,__@11}) ->
                                          {cont,{error,__@11 + 1}}
                                   end)
    of
        {error,__@12} ->
            error;
        __@13 ->
            __@13
    end.

handle_call(document, __from@1, _state@1) ->
    _document@1 = 'Elixir.Keyword':get(_state@1, document),
    {reply,_document@1,_state@1};
handle_call(ast, __from@1, _state@1) ->
    _ast@1 = 'Elixir.Keyword':get(_state@1, ast),
    {reply,_ast@1,_state@1};
handle_call(pid, __from@1, _state@1) ->
    {reply,self(),_state@1}.

handle_msg({system,{__@1,__@2},{terminate,__@4} = __@3},
           {__@5,__@6,__@7,__@8}) ->
    erlang:send(__@1, {__@2,ok}),
    apply(__@5, terminate, [__@4,__@8]);
handle_msg({'$gen_call',{__@2,__@3} = __@1,__@4}, {__@5,__@6,__@7,__@8}) ->
    case handle_call(__@4, __@1, __@8) of
        {reply,__@9,__@10} ->
            erlang:send(__@2, {__@3,__@9}),
            loop(__@5, __@6, __@7, __@10)
    end;
handle_msg({'EXIT',__@1,__@2}, {__@3,__@4,__@5,__@6}) ->
    case fetch_child_from_pid(__@5, __@1) of
        error ->
            exit(shutdown);
        {{__@7,__@8,__@9},__@10} ->
            {ok,__@11} = restart_child(__@8, __@9),
            __@12 = insert_child_at(__@5, {__@11,__@8,__@9}, __@10),
            loop(__@3, __@4, __@12, __@6)
    end.

init(_ast@1) ->
    lumen_intrinsics:println({document_supervisor, _ast@1}),
    %_window@1 =
    %    'Elixir.GenServer':call('Elixir.DbMonDemo.WindowSupervisor',
    %                            window, infinity),
    {ok,_window@1} = 'Elixir.Lumen.Web.Window':window(),
    {ok,_document@1} = 'Elixir.Lumen.Web.Window':document(_window@1),
    _children@1 = [{'Elixir.DbMonDemo.BodySupervisor',_ast@1}],
    lumen_intrinsics:println({document_supervisor_children, _children@1}),
    {ok,_children@1,[{document,_document@1},{ast,_ast@1}]}.

init_it(__@1, __@2, __@3) ->
    {ok,__@4,__@5} = apply(__@1, init, [__@3]),
    __@6 = sup_children(__@4),
    loop(__@1, __@2, __@6, __@5).

insert_child_at(__@1, __@2, __@3) ->
    'Elixir.List':insert_at(__@1, __@3, __@2).

loop(__@1, __@2, __@3, __@4) ->
    receive
        __@5 ->
            handle_msg(__@5, {__@1,__@2,__@3,__@4})
    end.

restart_child(__@1, __@2) ->
    {ok,{__@3,__@4,__@5}} =
        maps:find(start, apply(__@1, child_spec, [__@2])),
    apply(__@3, __@4, __@5).

start_link(_ast@1) ->
    'Elixir.DbMonDemo.Supervisor':start_link('Elixir.DbMonDemo.DocumentSupervisor',
                                             _ast@1,
                                             [{name,
                                               'Elixir.DbMonDemo.DocumentSupervisor'}]).

sup_children(__@1) ->
    element(1,
            'Elixir.Enum':reduce(__@1,
                                 {[],0},
                                 fun({__@2,__@3}, {__@4,__@5}) ->
                                        {ok,__@6} =
                                            restart_child(__@2, __@3),
                                        __@7 =
                                            insert_child_at(__@4,
                                                            {__@6,
                                                             __@2,
                                                             __@3},
                                                            __@5),
                                        {__@7,__@5 + 1}
                                 end)).

terminate(__@1, __@2) ->
    ok.

