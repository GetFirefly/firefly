-file("/home/build/elixir/lib/elixir/lib/range.ex", 1).

-module('Elixir.Range').

%-compile([no_auto_import,{inline,[{normalize,2}]}]).

-spec new(integer(), integer()) -> t().

-spec 'disjoint?'(t(), t()) -> boolean().

-export_type([t/2]).

%-type t(first, last) ::
%          #{'__struct__' := 'Elixir.Range',
%            first := first,
%            last := last}.

-export_type([t/0]).

-type t() ::
          #{'__struct__' := 'Elixir.Range',
            first := integer(),
            last := integer()}.

-export(['__info__'/1,
         '__struct__'/0,
         '__struct__'/1,
         'disjoint?'/2,
         new/2,
         'range?'/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Range';
'__info__'(functions) ->
    [{'__struct__',0},
     {'__struct__',1},
     {'disjoint?',2},
     {new,2},
     {'range?',1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Range', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Range', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Range', Key);
'__info__'(deprecated) ->
    [{{'range?',1},<<"Pattern match on first..last instead">>}].

'__struct__'() ->
    #{'__struct__' => 'Elixir.Range',first => nil,last => nil}.

'__struct__'(__@1) ->
    'Elixir.Enum':reduce(__@1,
                         #{'__struct__' => 'Elixir.Range',
                           first => nil,
                           last => nil},
                         fun({__@2,__@3}, __@4) ->
                                maps:update(__@2, __@3, __@4)
                         end).

'disjoint?'(#{'__struct__' := 'Elixir.Range',
              first := _first1@1,
              last := _last1@1},
            #{'__struct__' := 'Elixir.Range',
              first := _first2@1,
              last := _last2@1}) ->
    {_first1@2,_last1@2} = normalize(_first1@1, _last1@1),
    {_first2@2,_last2@2} = normalize(_first2@1, _last2@1),
    case _last2@2 < _first1@2 of
        false ->
            _last1@2 < _first2@2;
        true ->
            true
    end.

new(_first@1, _last@1)
    when
        is_integer(_first@1)
        andalso
        is_integer(_last@1) ->
    #{first => _first@1,last => _last@1,'__struct__' => 'Elixir.Range'};
new(_first@1, _last@1) ->
    error('Elixir.ArgumentError':exception(<<"ranges (first..last) expe"
                                             "ct both sides to be integ"
                                             "ers, ",
                                             "got: ",
                                             ('Elixir.Kernel':inspect(_first@1))/binary,
                                             "..",
                                             ('Elixir.Kernel':inspect(_last@1))/binary>>)).

normalize(_first@1, _last@1) when _first@1 > _last@1 ->
    {_last@1,_first@1};
normalize(_first@1, _last@1) ->
    {_first@1,_last@1}.

'range?'(#{'__struct__' := 'Elixir.Range',
           first := _first@1,
           last := _last@1})
    when
        is_integer(_first@1)
        andalso
        is_integer(_last@1) ->
    true;
'range?'(_) ->
    false.

