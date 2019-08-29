-file("lib/db_mon_demo/element_supervisor.ex", 1).

-module('Elixir.DbMonDemo.ElementSupervisor').

-export(['__info__'/1,
         build_child_from_node/2,
         build_children_from_ast/2,
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
    'Elixir.DbMonDemo.ElementSupervisor';
'__info__'(functions) ->
    [{build_child_from_node,2},
     {build_children_from_ast,2},
     {child_spec,1},
     {handle_call,3},
     {init,1},
     {init_it,3},
     {start_link,1},
     {terminate,2}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.DbMonDemo.ElementSupervisor', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.DbMonDemo.ElementSupervisor', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.DbMonDemo.ElementSupervisor', Key);
'__info__'(deprecated) ->
    [].

build_child_from_node({element,_tag_name@1,_attributes@1,_children@1},
                      _element@1) ->
    {'Elixir.DbMonDemo.ElementSupervisor',
     {_tag_name@1,_attributes@1,_children@1,_element@1}};
build_child_from_node({text,_value@1}, _element@1) ->
    {'Elixir.DbMonDemo.TextWorker',{_value@1,_element@1}}.

build_children_from_ast([], __element@1) ->
    [];
build_children_from_ast([_node@1|_nodes@1], _element@1) ->
    lumen_intrinsics:println({abc, _node@1, _nodes@1}),
    [build_child_from_node(_node@1, _element@1)|
     build_children_from_ast(_nodes@1, _element@1)].

child_spec(__@1) ->
    #{id => 'Elixir.DbMonDemo.ElementSupervisor',
      start => {'Elixir.DbMonDemo.ElementSupervisor',start_link,[__@1]},
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

handle_call(element, __from@1, _state@1) ->
    _element@1 = 'Elixir.Keyword':get(_state@1, element),
    {reply,_element@1,_state@1};
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

init({_tag_name@1,_attributes@1,_ast@1,_parent_node@1}) ->
    lumen_intrinsics:println({tag_name, _tag_name@1}),
    lumen_intrinsics:println({attributes, _attributes@1}),
    lumen_intrinsics:println({ast, _ast@1}),
    lumen_intrinsics:println({parent_node, _parent_node@1}),
    lumen_intrinsics:println({aaa, _tag_name@1,_attributes@1,_ast@1,_parent_node@1}),
    {ok,_element@1} = init_element(_tag_name@1, _attributes@1),
    {ok,_element@2} =
        'Elixir.Lumen.Web.Node':append_child(_parent_node@1, _element@1),
    _children@1 = build_children_from_ast(_ast@1, _element@2),
    {ok,_children@1,[{element,_element@2},{ast,_ast@1}]}.

init_element(_tag_name@1, _attributes@1) ->
    _document@1 =
        'Elixir.GenServer':call('Elixir.DbMonDemo.DocumentSupervisor',
                                document),
    {ok,_element@1} =
        'Elixir.Lumen.Web.Document':create_element(_document@1,
                                                   _tag_name@1),
    'Elixir.Enum':each(_attributes@1,
                       fun([attribute,_name@1,_value@1]) ->
                              'Elixir.Lumen.Web.Element':set_attribute(_element@1,
                                                                       _name@1,
                                                                       _value@1)
                       end),
    {ok,_element@1}.

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

start_link(_element@1) ->
    'Elixir.DbMonDemo.Supervisor':start_link('Elixir.DbMonDemo.ElementSupervisor',
                                             _element@1).

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

