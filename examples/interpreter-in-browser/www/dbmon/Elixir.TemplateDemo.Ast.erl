-file("lib/template_demo/ast.ex", 1).

-module('Elixir.TemplateDemo.Ast').

-compile([no_auto_import]).

-export(['__info__'/1,ast/0]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.TemplateDemo.Ast';
'__info__'(functions) ->
    [{ast,0}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.TemplateDemo.Ast', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.TemplateDemo.Ast', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.TemplateDemo.Ast', Key);
'__info__'(deprecated) ->
    [].

%ast() ->
%      [{text,<<"\n  ">>},
%       {element,p,[],[{text,<<"Hello">>}]},
%       {text,<<"\n  ">>},
%       {element,
%        h1,
%        [],
%        [{text,<<"This is an example rendering from Lumen">>}]},
%       {text,<<"\n">>}].

ast() ->
    [{element,
      <<"div">>,
      [{attribute,<<"id">>,<<"foo">>}],
      [{text,<<"\n  ">>},
       {element,<<"p">>,[],[{text,<<"Hello">>}]},
       {text,<<"\n  ">>},
       {element,
        <<"h1">>,
        [],
        [{text,<<"This is an example rendering from Lumen">>}]},
       {text,<<"\n">>}]}].



%ast() ->
%    [{element,
%      'div',
%      [{attribute,id,<<"foo">>}],
%      [{text,<<"\n  ">>},
%       {element,p,[],[]},
%       {text,<<"\n  ">>},
%       {element,
%        h1,
%        [],
%        []},
%       {text,<<"\n">>}]}].
%
