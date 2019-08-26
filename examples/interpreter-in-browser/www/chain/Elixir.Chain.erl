-file("lib/chain.ex", 33).

-module('Elixir.Chain').

-export(['__info__'/1,console/1,counter/2,create_processes/2,dom/1]).

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
    [{console,1},{counter,2},{create_processes,2},{dom,1}];
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

console(_n@1) ->
    run(_n@1, fun console_output/1).

console_output(_text@1) ->
    'lumen_intrinsics':println(<<case self() of
                           _rewrite@1 when is_binary(_rewrite@1) ->
                               _rewrite@1;
                           _rewrite@1 ->
                               'lumen_intrinsics':format(_rewrite@1)
                       end/binary,
                       " ",
                       case _text@1 of
                           _rewrite@2 when is_binary(_rewrite@2) ->
                               _rewrite@2;
                           _rewrite@2 ->
                               'lumen_intrinsics':format(_rewrite@2)
                       end/binary>>).

counter(_next_pid@1, _output@1) ->
    _output@1(<<"spawned">>),
    receive
        _n@1 ->
            _output@1(<<"received ",
                        case _n@1 of
                            _rewrite@1 when is_binary(_rewrite@1) ->
                                _rewrite@1;
                            _rewrite@1 ->
                                'lumen_intrinsics':format(_rewrite@1)
                        end/binary>>),
            _sent@1 = _n@1 + 1,
            erlang:send(_next_pid@1, _sent@1),
            _output@1(<<"sent ",
                        case _sent@1 of
                            _rewrite@2 when is_binary(_rewrite@2) ->
                                _rewrite@2;
                            _rewrite@2 ->
                                'lumen_intrinsics':format(_rewrite@2)
                        end/binary,
                        " to ",
                        case _next_pid@1 of
                            _rewrite@3 when is_binary(_rewrite@3) ->
                                _rewrite@3;
                            _rewrite@3 ->
                                'lumen_intrinsics':format(_rewrite@3)
                        end/binary>>)
    end.

create_processes(_n@1, _output@1) ->
    _last@1 =
        'Elixir.Enum':reduce('Elixir.Range':new(1, _n@1),
                             self(),
                             fun(_, _send_to@1) ->
                                    spawn('Elixir.Chain',
                                          counter,
                                          [_send_to@1,_output@1])
                             end),
    erlang:send(_last@1, 0),
    receive
        _final_answer@1 when is_integer(_final_answer@1) ->
            <<"Result is ",
              ('lumen_intrinsics':format(_final_answer@1))/binary>>,
            _final_answer@1
    end.

dom(_n@1) ->
    run(_n@1, fun dom_output/1).

dom_output(_text@1) ->
    {ok, _window@1} = 'Elixir.Lumen.Web.Window':window(),
    {ok, _document@1} = 'Elixir.Lumen.Web.Window':document(_window@1),
    {ok,_tr@1} =
        'Elixir.Lumen.Web.Document':create_element(_document@1,
                                                   <<"tr">>),
    _pid_text@1 =
        'Elixir.Lumen.Web.Document':create_text_node(_document@1,
                                                     case self() of
                                                         _rewrite@1
                                                             when
                                                                 is_binary(_rewrite@1) ->
                                                             _rewrite@1;
                                                         _rewrite@1 ->
                                                             'lumen_intrinsics':format(_rewrite@1)
                                                     end),
    {ok,_pid_td@1} =
        'Elixir.Lumen.Web.Document':create_element(_document@1,
                                                   <<"td">>),
    'Elixir.Lumen.Web.Node':append_child(_pid_td@1, _pid_text@1),
    'Elixir.Lumen.Web.Node':append_child(_tr@1, _pid_td@1),
    _text_text@1 =
        'Elixir.Lumen.Web.Document':create_text_node(_document@1,
                                                     case _text@1 of
                                                         _rewrite@2
                                                             when
                                                                 is_binary(_rewrite@2) ->
                                                             _rewrite@2;
                                                         _rewrite@2 ->
                                                             'lumen_intrinsics':format(_rewrite@2)
                                                     end),
    {ok,_text_td@1} =
        'Elixir.Lumen.Web.Document':create_element(_document@1,
                                                   <<"td">>),
    'Elixir.Lumen.Web.Node':append_child(_text_td@1, _text_text@1),
    'Elixir.Lumen.Web.Node':append_child(_tr@1, _text_td@1),
    {ok,_output@1} =
        'Elixir.Lumen.Web.Document':get_element_by_id(_document@1, <<"output">>),
    'Elixir.Lumen.Web.Node':append_child(_output@1, _tr@1).

run(_n@1, _output@1) ->
    {_time@1,_value@1} =
        timer:tc('Elixir.Chain', create_processes, [_n@1,_output@1]),
    _output@1(<<"Chain.run(",
                case _n@1 of
                    _rewrite@1 when is_binary(_rewrite@1) ->
                        _rewrite@1;
                    _rewrite@1 ->
                        'lumen_intrinsics':format(_rewrite@1)
                end/binary,
                ") in ",
                case _time@1 of
                    _rewrite@2 when is_binary(_rewrite@2) ->
                        _rewrite@2;
                    _rewrite@2 ->
                        'lumen_intrinsics':format(_rewrite@2)
                end/binary,
                " microseconds">>),
    _value@1.

