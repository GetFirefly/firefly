-module('Elixir.ExSample').

-export(['__info__'/1, run_me/1]).

-spec '__info__'(attributes | compile | functions |
		 macros | md5 | module | deprecated) -> any().

'__info__'(module) -> 'Elixir.ExSample';
'__info__'(functions) -> [{run_me, 1}];
'__info__'(macros) -> [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.ExSample', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.ExSample', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.ExSample', Key);
'__info__'(deprecated) -> [].

run_me(_arg@1) ->
    lumen_intrinsics:println(<<"Doing the Lumen web work...">>),
    {ok, _window@1} = 'Elixir.Lumen.Web.Window':window(),
    {ok, _document@1} =
	'Elixir.Lumen.Web.Window':document(_window@1),
    {ok, _paragraph@1} =
	'Elixir.Lumen.Web.Document':create_element(_document@1,
						   <<"p">>),
    _text@1 =
	'Elixir.Lumen.Web.Document':create_text_node(_document@1,
						     <<"This text was created through Elixir "
						       "in your browser.">>),
    ok = 'Elixir.Lumen.Web.Node':append_child(_paragraph@1,
					      _text@1),
    {ok, _output@1} =
	'Elixir.Lumen.Web.Document':get_element_by_id(_document@1,
						      <<"output">>),
    ok = 'Elixir.Lumen.Web.Node':append_child(_output@1,
					      _paragraph@1),
    lumen_intrinsics:println(<<"Done!">>).
