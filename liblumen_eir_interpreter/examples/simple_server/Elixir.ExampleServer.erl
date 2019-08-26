-file("lib/example_server.ex", 1).

-module('Elixir.ExampleServer').

-behaviour('Elixir.GenServer').

-export(['__info__'/1,
         child_spec/1,
         code_change/3,
         handle_call/3,
         handle_cast/2,
         handle_info/2,
         init/1,
         pop/1,
         push/2,
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
    'Elixir.ExampleServer';
'__info__'(functions) ->
    [{child_spec,1},
     {code_change,3},
     {handle_call,3},
     {handle_cast,2},
     {handle_info,2},
     {init,1},
     {pop,1},
     {push,2},
     {start_link,1},
     {terminate,2}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.ExampleServer', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.ExampleServer', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.ExampleServer', Key);
'__info__'(deprecated) ->
    [].

handle_call(pop, __from@1, [_head@1|_tail@1]) ->
    {reply,_head@1,_tail@1}.

handle_cast({push,_element@1}, _state@1) ->
    {noreply,[_element@1 | _state@1]}.

init(_stack@1) ->
    {ok,_stack@1}.

pop(_pid@1) ->
    'Elixir.GenServer':call(_pid@1, pop).

push(_pid@1, _element@1) ->
    'Elixir.GenServer':cast(_pid@1, {push,_element@1}).

start_link(_default@1) when is_list(_default@1) ->
    'Elixir.GenServer':start_link('Elixir.ExampleServer', _default@1).

-file("lib/gen_server.ex", 729).

child_spec(__@1) ->
    __@2 =
        #{id => 'Elixir.ExampleServer',
          start => {'Elixir.ExampleServer',start_link,[__@1]}},
    'Elixir.Supervisor':child_spec(__@2, []).

-file("lib/gen_server.ex", 798).

code_change(__@1, __@2, __@3) ->
    {ok,__@2}.

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
    error_logger:error_msg(__@5, ['Elixir.ExampleServer',__@4,__@1]),
    {noreply,__@2}.

-file("lib/gen_server.ex", 793).

terminate(__@1, __@2) ->
    ok.

