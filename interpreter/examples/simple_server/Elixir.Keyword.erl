-file("/home/build/elixir/lib/elixir/lib/keyword.ex", 1).

-module('Elixir.Keyword').

-compile([inline_list_funcs,{inline,[{delete,2}]}]).

-spec 'equal?'(t(), t()) -> boolean().

-spec new() -> [].

-spec put_new(t(), key(), value()) -> t().

-spec drop(t(), [key()]) -> t().

-spec get(t(), key(), value()) -> value().

-spec split(t(), [key()]) -> {t(), t()}.

-spec merge(t(), t(), fun((key(), value(), value()) -> value())) -> t().

-spec 'has_key?'(t(), key()) -> boolean().

-spec pop_lazy(t(), key(), fun(() -> value())) -> {value(), t()}.

-spec pop(t(), key(), value()) -> {value(), t()}.

-spec get_values(t(), key()) -> [value()].

-spec 'keyword?'(term()) -> boolean().

-spec put_new_lazy(t(), key(), fun(() -> value())) -> t().

-spec keys(t()) -> [key()].

-spec fetch(t(), key()) -> {ok, value()} | error.

-spec delete(t(), key()) -> t().

-spec get_and_update(t(), key(), fun((value()) -> {get, value()} | pop)) ->
                        {get, t()}
                        when get :: term().

-spec take(t(), [key()]) -> t().

-spec delete(t(), key(), value()) -> t().

-spec update(t(), key(), value(), fun((value()) -> value())) -> t().

-spec pop_first(t(), key(), value()) -> {value(), t()}.

-spec 'update!'(t(), key(), fun((value()) -> value())) -> t().

-spec new('Elixir.Enum':t()) -> t().

-spec 'replace!'(t(), key(), value()) -> t().

-spec get_lazy(t(), key(), fun(() -> value())) -> value().

-spec delete_first(t(), key()) -> t().

-spec new('Elixir.Enum':t(), fun((term()) -> {key(), value()})) -> t().

-spec merge(t(), t()) -> t().

-spec values(t()) -> [value()].

-spec 'fetch!'(t(), key()) -> value().

-spec 'get_and_update!'(t(), key(), fun((value()) -> {get, value()})) ->
                           {get, t()}
                           when get :: term().

-spec put(t(), key(), value()) -> t().

-spec to_list(t()) -> t().

-export_type([t/1]).

-type t(value) :: [{key(), value}].

-export_type([t/0]).

-type t() :: [{key(), value()}].

-export_type([value/0]).

-type value() :: any().

-export_type([key/0]).

-type key() :: atom().

-export(['__info__'/1,
         delete/2,
         delete/3,
         delete_first/2,
         drop/2,
         'equal?'/2,
         fetch/2,
         'fetch!'/2,
         get/2,
         get/3,
         get_and_update/3,
         'get_and_update!'/3,
         get_lazy/3,
         get_values/2,
         'has_key?'/2,
         keys/1,
         'keyword?'/1,
         merge/2,
         merge/3,
         new/0,
         new/1,
         new/2,
         pop/2,
         pop/3,
         pop_first/2,
         pop_first/3,
         pop_lazy/3,
         put/3,
         put_new/3,
         put_new_lazy/3,
         replace/3,
         'replace!'/3,
         size/1,
         split/2,
         take/2,
         to_list/1,
         update/4,
         'update!'/3,
         values/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Keyword';
'__info__'(functions) ->
    [{delete,2},
     {delete,3},
     {delete_first,2},
     {drop,2},
     {'equal?',2},
     {fetch,2},
     {'fetch!',2},
     {get,2},
     {get,3},
     {get_and_update,3},
     {'get_and_update!',3},
     {get_lazy,3},
     {get_values,2},
     {'has_key?',2},
     {keys,1},
     {'keyword?',1},
     {merge,2},
     {merge,3},
     {new,0},
     {new,1},
     {new,2},
     {pop,2},
     {pop,3},
     {pop_first,2},
     {pop_first,3},
     {pop_lazy,3},
     {put,3},
     {put_new,3},
     {put_new_lazy,3},
     {replace,3},
     {'replace!',3},
     {size,1},
     {split,2},
     {take,2},
     {to_list,1},
     {update,4},
     {'update!',3},
     {values,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Keyword', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Keyword', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Keyword', Key);
'__info__'(deprecated) ->
    [{{replace,3},<<"Use Keyword.fetch/2 + Keyword.put/3 instead">>},
     {{size,1},<<"Use Kernel.length/1 instead">>}].

delete(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keymember(_key@1, 1, _keywords@1) of
        true ->
            delete_key(_keywords@1, _key@1);
        _ ->
            _keywords@1
    end.

delete(_keywords@1, _key@1, _value@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keymember(_key@1, 1, _keywords@1) of
        true ->
            delete_key_value(_keywords@1, _key@1, _value@1);
        _ ->
            _keywords@1
    end.

delete_first(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keymember(_key@1, 1, _keywords@1) of
        true ->
            delete_first_key(_keywords@1, _key@1);
        _ ->
            _keywords@1
    end.

delete_first_key([{_key@1,_}|_tail@1], _key@1) ->
    _tail@1;
delete_first_key([{_,_} = _pair@1|_tail@1], _key@1) ->
    [_pair@1|delete_first_key(_tail@1, _key@1)];
delete_first_key([], __key@1) ->
    [].

delete_key([{_key@1,_}|_tail@1], _key@1) ->
    delete_key(_tail@1, _key@1);
delete_key([{_,_} = _pair@1|_tail@1], _key@1) ->
    [_pair@1|delete_key(_tail@1, _key@1)];
delete_key([], __key@1) ->
    [].

delete_key_value([{_key@1,_value@1}|_tail@1], _key@1, _value@1) ->
    delete_key_value(_tail@1, _key@1, _value@1);
delete_key_value([{_,_} = _pair@1|_tail@1], _key@1, _value@1) ->
    [_pair@1|delete_key_value(_tail@1, _key@1, _value@1)];
delete_key_value([], __key@1, __value@1) ->
    [].

do_merge([{_key@1,_value2@1}|_tail@1],
         _acc@1,
         _rest@1,
         _original@1,
         _fun@1,
         _keywords2@1)
    when is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _original@1) of
        {_key@1,_value1@1} ->
            _acc@2 =
                [{_key@1,_fun@1(_key@1, _value1@1, _value2@1)}|_acc@1],
            _original@2 = lists:keydelete(_key@1, 1, _original@1),
            do_merge(_tail@1,
                     _acc@2,
                     delete(_rest@1, _key@1),
                     _original@2,
                     _fun@1,
                     _keywords2@1);
        false ->
            do_merge(_tail@1,
                     [{_key@1,_value2@1}|_acc@1],
                     _rest@1,
                     _original@1,
                     _fun@1,
                     _keywords2@1)
    end;
do_merge([], _acc@1, _rest@1, __original@1, __fun@1, __keywords2@1) ->
    _rest@1 ++ lists:reverse(_acc@1);
do_merge(__other@1,
         __acc@1,
         __rest@1,
         __original@1,
         __fun@1,
         _keywords2@1) ->
    error('Elixir.ArgumentError':exception(<<"expected a keyword list a"
                                             "s the second argument, go"
                                             "t: ",
                                             ('Elixir.Kernel':inspect(_keywords2@1))/binary>>)).

drop(_keywords@1, _keys@1) when is_list(_keywords@1) ->
    lists:filter(fun({_key@1,_}) ->
                        not 'Elixir.Enum':'member?'(_keys@1, _key@1)
                 end,
                 _keywords@1).

'equal?'(_left@1, _right@1)
    when
        is_list(_left@1)
        andalso
        is_list(_right@1) ->
    lists:sort(_left@1) == lists:sort(_right@1).

fetch(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_value@1} ->
            {ok,_value@1};
        false ->
            error
    end.

'fetch!'(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_value@1} ->
            _value@1;
        false ->
            error('Elixir.KeyError':exception([{key,_key@1},
                                               {term,_keywords@1}]))
    end.

get(__@1, __@2) ->
    get(__@1, __@2, nil).

get(_keywords@1, _key@1, _default@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_value@1} ->
            _value@1;
        false ->
            _default@1
    end.

get_and_update(_keywords@1, _key@1, _fun@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    get_and_update(_keywords@1, [], _key@1, _fun@1).

get_and_update([{_key@1,_current@1}|_t@1], _acc@1, _key@1, _fun@1) ->
    case _fun@1(_current@1) of
        {_get@1,_value@1} ->
            {_get@1,lists:reverse(_acc@1, [{_key@1,_value@1}|_t@1])};
        pop ->
            {_current@1,lists:reverse(_acc@1, _t@1)};
        _other@1 ->
            error('Elixir.RuntimeError':exception(<<"the given function"
                                                    " must return a two"
                                                    "-element tuple or "
                                                    ":pop, got: ",
                                                    ('Elixir.Kernel':inspect(_other@1))/binary>>))
    end;
get_and_update([{_,_} = _h@1|_t@1], _acc@1, _key@1, _fun@1) ->
    get_and_update(_t@1, [_h@1|_acc@1], _key@1, _fun@1);
get_and_update([], _acc@1, _key@1, _fun@1) ->
    case _fun@1(nil) of
        {_get@1,_update@1} ->
            {_get@1,[{_key@1,_update@1}|lists:reverse(_acc@1)]};
        pop ->
            {nil,lists:reverse(_acc@1)};
        _other@1 ->
            error('Elixir.RuntimeError':exception(<<"the given function"
                                                    " must return a two"
                                                    "-element tuple or "
                                                    ":pop, got: ",
                                                    ('Elixir.Kernel':inspect(_other@1))/binary>>))
    end.

'get_and_update!'(_keywords@1, _key@1, _fun@1) ->
    'get_and_update!'(_keywords@1, _key@1, _fun@1, []).

'get_and_update!'([{_key@1,_value@1}|_keywords@1],
                  _key@1,
                  _fun@1,
                  _acc@1) ->
    case _fun@1(_value@1) of
        {_get@1,_value@2} ->
            {_get@1,
             lists:reverse(_acc@1,
                           [{_key@1,_value@2}|
                            delete(_keywords@1, _key@1)])};
        pop ->
            {_value@1,lists:reverse(_acc@1, _keywords@1)};
        _other@1 ->
            error('Elixir.RuntimeError':exception(<<"the given function"
                                                    " must return a two"
                                                    "-element tuple or "
                                                    ":pop, got: ",
                                                    ('Elixir.Kernel':inspect(_other@1))/binary>>))
    end;
'get_and_update!'([{_,_} = _e@1|_keywords@1], _key@1, _fun@1, _acc@1) ->
    'get_and_update!'(_keywords@1, _key@1, _fun@1, [_e@1|_acc@1]);
'get_and_update!'([], _key@1, __fun@1, _acc@1) when is_atom(_key@1) ->
    error('Elixir.KeyError':exception([{key,_key@1},{term,_acc@1}])).

get_lazy(_keywords@1, _key@1, _fun@1)
    when
        (is_list(_keywords@1)
         andalso
         is_atom(_key@1))
        andalso
        is_function(_fun@1, 0) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_value@1} ->
            _value@1;
        false ->
            _fun@1()
    end.

get_values(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    _fun@1 =
        fun({_key@2,_val@1}) when _key@1 =:= _key@2 ->
               {true,_val@1};
           ({_,_}) ->
               false
        end,
    lists:filtermap(_fun@1, _keywords@1).

'has_key?'(_keywords@1, _key@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    lists:keymember(_key@1, 1, _keywords@1).

keys(_keywords@1) when is_list(_keywords@1) ->
    lists:map(fun({_k@1,_}) ->
                     _k@1
              end,
              _keywords@1).

'keyword?'([{_key@1,__value@1}|_rest@1]) when is_atom(_key@1) ->
    'keyword?'(_rest@1);
'keyword?'([]) ->
    true;
'keyword?'(__other@1) ->
    false.

merge(_keywords1@1, []) when is_list(_keywords1@1) ->
    _keywords1@1;
merge([], _keywords2@1) when is_list(_keywords2@1) ->
    _keywords2@1;
merge(_keywords1@1, _keywords2@1)
    when
        is_list(_keywords1@1)
        andalso
        is_list(_keywords2@1) ->
    case 'keyword?'(_keywords2@1) of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            error('Elixir.ArgumentError':exception(<<"expected a keywor"
                                                     "d list as the sec"
                                                     "ond argument, got"
                                                     ": ",
                                                     ('Elixir.Kernel':inspect(_keywords2@1))/binary>>));
        _ ->
            _fun@1 =
                fun({_key@1,__value@1}) when is_atom(_key@1) ->
                       not 'has_key?'(_keywords2@1, _key@1);
                   (_) ->
                       error('Elixir.ArgumentError':exception(<<"expect"
                                                                "ed a k"
                                                                "eyword"
                                                                " list "
                                                                "as the"
                                                                " first"
                                                                " argum"
                                                                "ent, g"
                                                                "ot: ",
                                                                ('Elixir.Kernel':inspect(_keywords1@1))/binary>>))
                end,
            lists:filter(_fun@1, _keywords1@1) ++ _keywords2@1
    end.

merge(_keywords1@1, _keywords2@1, _fun@1)
    when
        (is_list(_keywords1@1)
         andalso
         is_list(_keywords2@1))
        andalso
        is_function(_fun@1, 3) ->
    case 'keyword?'(_keywords1@1) of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            error('Elixir.ArgumentError':exception(<<"expected a keywor"
                                                     "d list as the fir"
                                                     "st argument, got:"
                                                     " ",
                                                     ('Elixir.Kernel':inspect(_keywords1@1))/binary>>));
        _ ->
            do_merge(_keywords2@1,
                     [],
                     _keywords1@1,
                     _keywords1@1,
                     _fun@1,
                     _keywords2@1)
    end.

new() ->
    [].

new(_pairs@1) ->
    new(_pairs@1,
        fun(_pair@1) ->
               _pair@1
        end).

new(_pairs@1, _transform@1) ->
    _fun@1 =
        fun(_el@1, _acc@1) ->
               {_k@1,_v@1} = _transform@1(_el@1),
               put_new(_acc@1, _k@1, _v@1)
        end,
    lists:foldl(_fun@1, [], 'Elixir.Enum':reverse(_pairs@1)).

pop(__@1, __@2) ->
    pop(__@1, __@2, nil).

pop(_keywords@1, _key@1, _default@1) when is_list(_keywords@1) ->
    case fetch(_keywords@1, _key@1) of
        {ok,_value@1} ->
            {_value@1,delete(_keywords@1, _key@1)};
        error ->
            {_default@1,_keywords@1}
    end.

pop_first(__@1, __@2) ->
    pop_first(__@1, __@2, nil).

pop_first(_keywords@1, _key@1, _default@1) when is_list(_keywords@1) ->
    case lists:keytake(_key@1, 1, _keywords@1) of
        {value,{_key@1,_value@1},_rest@1} ->
            {_value@1,_rest@1};
        false ->
            {_default@1,_keywords@1}
    end.

pop_lazy(_keywords@1, _key@1, _fun@1)
    when
        is_list(_keywords@1)
        andalso
        is_function(_fun@1, 0) ->
    case fetch(_keywords@1, _key@1) of
        {ok,_value@1} ->
            {_value@1,delete(_keywords@1, _key@1)};
        error ->
            {_fun@1(),_keywords@1}
    end.

put(_keywords@1, _key@1, _value@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    [{_key@1,_value@1}|delete(_keywords@1, _key@1)].

put_new(_keywords@1, _key@1, _value@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_} ->
            _keywords@1;
        false ->
            [{_key@1,_value@1}|_keywords@1]
    end.

put_new_lazy(_keywords@1, _key@1, _fun@1)
    when
        (is_list(_keywords@1)
         andalso
         is_atom(_key@1))
        andalso
        is_function(_fun@1, 0) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_} ->
            _keywords@1;
        false ->
            [{_key@1,_fun@1()}|_keywords@1]
    end.

replace(_keywords@1, _key@1, _value@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_} ->
            [{_key@1,_value@1}|delete(_keywords@1, _key@1)];
        false ->
            _keywords@1
    end.

'replace!'(_keywords@1, _key@1, _value@1)
    when
        is_list(_keywords@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _keywords@1) of
        {_key@1,_} ->
            [{_key@1,_value@1}|delete(_keywords@1, _key@1)];
        false ->
            error('Elixir.KeyError':exception([{key,_key@1},
                                               {term,_keywords@1}]))
    end.

size(_keyword@1) ->
    length(_keyword@1).

split(_keywords@1, _keys@1) when is_list(_keywords@1) ->
    _fun@1 =
        fun({_k@1,_v@1}, {_take@1,_drop@1}) ->
               case 'Elixir.Enum':'member?'(_keys@1, _k@1) of
                   true ->
                       {[{_k@1,_v@1}|_take@1],_drop@1};
                   false ->
                       {_take@1,[{_k@1,_v@1}|_drop@1]}
               end
        end,
    _acc@1 = {[],[]},
    {_take@2,_drop@2} = lists:foldl(_fun@1, _acc@1, _keywords@1),
    {lists:reverse(_take@2),lists:reverse(_drop@2)}.

take(_keywords@1, _keys@1) when is_list(_keywords@1) ->
    lists:filter(fun({_k@1,_}) ->
                        'Elixir.Enum':'member?'(_keys@1, _k@1)
                 end,
                 _keywords@1).

to_list(_keyword@1) when is_list(_keyword@1) ->
    _keyword@1.

update([{_key@1,_value@1}|_keywords@1], _key@1, __initial@1, _fun@1) ->
    [{_key@1,_fun@1(_value@1)}|delete(_keywords@1, _key@1)];
update([{_,_} = _e@1|_keywords@1], _key@1, _initial@1, _fun@1) ->
    [_e@1|update(_keywords@1, _key@1, _initial@1, _fun@1)];
update([], _key@1, _initial@1, __fun@1) when is_atom(_key@1) ->
    [{_key@1,_initial@1}].

'update!'(_keywords@1, _key@1, _fun@1) ->
    'update!'(_keywords@1, _key@1, _fun@1, _keywords@1).

'update!'([{_key@1,_value@1}|_keywords@1], _key@1, _fun@1, __dict@1) ->
    [{_key@1,_fun@1(_value@1)}|delete(_keywords@1, _key@1)];
'update!'([{_,_} = _e@1|_keywords@1], _key@1, _fun@1, _dict@1) ->
    [_e@1|'update!'(_keywords@1, _key@1, _fun@1, _dict@1)];
'update!'([], _key@1, __fun@1, _dict@1) when is_atom(_key@1) ->
    error('Elixir.KeyError':exception([{key,_key@1},{term,_dict@1}])).

values(_keywords@1) when is_list(_keywords@1) ->
    lists:map(fun({_,_v@1}) ->
                     _v@1
              end,
              _keywords@1).

