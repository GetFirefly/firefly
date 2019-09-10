-file("lib/chain.ex", 1).

-module('Elixir.Chain').

-compile([no_auto_import]).

-export(['__info__'/1,counter/1,create_processes/1,run/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Chain';
'__info__'(functions) ->
    [{counter,1},{create_processes,1},{run,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Chain', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Chain', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Chain', Key);
'__info__'(deprecated) ->
    [].

counter(_next_pid@1) ->
    receive
        _n@1 ->
            erlang:send(_next_pid@1, _n@1 + 1)
    end.

create_processes(_n@1) ->
    _last@1 =
        'Elixir.Enum':reduce('Elixir.Range':new(1, _n@1),
                             self(),
                             fun(_, _send_to@1) ->
                                    spawn('Elixir.Chain',
                                          counter,
                                          [_send_to@1])
                             end),
    erlang:send(_last@1, 0),
    receive
        _final_answer@1 when is_integer(_final_answer@1) ->
            <<"Result is ",
              ('Elixir.Kernel':inspect(_final_answer@1))/binary>>
    end.

run(_n@1) ->
    'Elixir.IO':puts('Elixir.Kernel':inspect(timer:tc('Elixir.Chain',
                                                      create_processes,
                                                      [_n@1]))).

