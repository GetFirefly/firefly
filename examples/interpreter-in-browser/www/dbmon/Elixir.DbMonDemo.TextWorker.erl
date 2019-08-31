-file("lib/db_mon_demo/text_worker.ex", 1).

-module('Elixir.DbMonDemo.TextWorker').

-behaviour('Elixir.GenServer').

-export(['__info__'/1,
         child_spec/1,
         code_change/3,
         handle_call/3,
         handle_cast/2,
         handle_info/2,
         init/1,
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
    'Elixir.DbMonDemo.TextWorker';
'__info__'(functions) ->
    [{child_spec,1},
     {code_change,3},
     {handle_call,3},
     {handle_cast,2},
     {handle_info,2},
     {init,1},
     {start_link,1},
     {terminate,2}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.DbMonDemo.TextWorker', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.DbMonDemo.TextWorker', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.DbMonDemo.TextWorker', Key);
'__info__'(deprecated) ->
    [].

init({_value@1,_parent@1}) ->
    _document@1 =
        'Elixir.GenServer':call('Elixir.DbMonDemo.DocumentSupervisor',
                                document, infinity),
    _element@1 =
        'Elixir.Lumen.Web.Document':create_text_node(_document@1,
                                                     _value@1),
    ok =
        'Elixir.Lumen.Web.Node':append_child(_parent@1, _element@1),
    {ok,[{element,_element@1}]}.

start_link(_args@1) ->
    'Elixir.GenServer':start_link('Elixir.DbMonDemo.TextWorker',
                                  _args@1).

-file("lib/gen_server.ex", 729).

child_spec(__@1) ->
    __@2 =
        #{id => 'Elixir.DbMonDemo.TextWorker',
          start => {'Elixir.DbMonDemo.TextWorker',start_link,[__@1]}},
    'Elixir.Supervisor':child_spec(__@2, []).

-file("lib/gen_server.ex", 798).

code_change(__@1, __@2, __@3) ->
    {ok,__@2}.

-file("lib/gen_server.ex", 744).

handle_call(__@1, __@2, __@3) ->
    __@5 =
        case 'Elixir.Process':info(self(), registered_name) of
            {_,[]} ->
                self();
            {_,__@4} ->
                __@4
        end,
    case erlang:phash2(1, 1) of
        0 ->
            error('Elixir.RuntimeError':exception(<<"attempted to call "
                                                    "GenServer ",
                                                    ('Elixir.Kernel':inspect(__@5))/binary,
                                                    " but no handle_cal"
                                                    "l/3 clause was pro"
                                                    "vided">>));
        1 ->
            {stop,{bad_call,__@1},__@3}
    end.

-file("lib/gen_server.ex", 775).

handle_cast(__@1, __@2) ->
    __@4 =
        case 'Elixir.Process':info(self(), registered_name) of
            {_,[]} ->
                self();
            {_,__@3} ->
                __@3
        end,
    case erlang:phash2(1, 1) of
        0 ->
            error('Elixir.RuntimeError':exception(<<"attempted to cast "
                                                    "GenServer ",
                                                    ('Elixir.Kernel':inspect(__@4))/binary,
                                                    " but no handle_cas"
                                                    "t/2 clause was pro"
                                                    "vided">>));
        1 ->
            {stop,{bad_cast,__@1},__@2}
    end.

-file("lib/gen_server.ex", 762).

handle_info(__@1, __@2) ->
    __@4 =
        case 'Elixir.Process':info(self(), registered_name) of
            {_,[]} ->
                self();
            {_,__@3} ->
                __@3
        end,
    __@5 =
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
         110],
    error_logger:error_msg(__@5,
                           ['Elixir.DbMonDemo.TextWorker',__@4,__@1]),
    {noreply,__@2}.

-file("lib/gen_server.ex", 793).

terminate(__@1, __@2) ->
    ok.

