-file("/home/build/elixir/lib/elixir/lib/access.ex", 1).

-module('Elixir.Access').

-callback pop(data, key()) -> {value(), data}
                 when data :: container() | any_container().

-callback get_and_update(data,
                         key(),
                         fun((value()) -> {get_value, value()} | pop)) ->
                            {get_value, data}
                            when data :: container() | any_container().

-callback fetch(term :: t(), key()) -> {ok, value()} | error.

-spec pop(data, key()) -> {value(), data} when data :: container().

-spec 'key!'(key()) ->
                access_fun(data :: elixir:struct() | map(),
                           get_value :: term()).

-spec key(key(), term()) ->
             access_fun(data :: elixir:struct() | map(),
                        get_value :: term()).

-spec get_and_update(data,
                     key(),
                     fun((value()) -> {get_value, value()} | pop)) ->
                        {get_value, data}
                        when data :: container().

-spec get(container(), term(), term()) -> term();
         (nil_container(), any(), default) -> default.

-spec filter(fun((term()) -> boolean())) ->
                access_fun(data :: list(), get_value :: list()).

-spec fetch(container(), term()) -> {ok, term()} | error;
           (nil_container(), any()) -> error.

-spec elem(non_neg_integer()) ->
              access_fun(data :: tuple(), get_value :: term()).

-spec at(non_neg_integer()) ->
            access_fun(data :: list(), get_value :: term()).

-spec all() -> access_fun(data :: list(), get_value :: list()).

-export_type([access_fun/2]).

-type access_fun(data, get_value) ::
          get_fun(data, get_value) | get_and_update_fun(data, get_value).

-export_type([get_and_update_fun/2]).

-type get_and_update_fun(data, get_value) ::
          fun((get_and_update, data, fun((term()) -> term())) ->
                  {get_value, new_data :: container()} | pop).

-export_type([get_fun/2]).

-type get_fun(data, get_value) ::
          fun((get, data, fun((term()) -> term())) ->
                  {get_value, new_data :: container()}).

-export_type([value/0]).

-type value() :: any().

-export_type([key/0]).

-type key() :: any().

-export_type([t/0]).

-type t() :: container() | nil_container() | any_container().

-export_type([any_container/0]).

-type any_container() :: any().

-export_type([nil_container/0]).

-type nil_container() :: nil.

-export_type([container/0]).

-type container() :: elixir:keyword() | elixir:struct() | map().

-export(['__info__'/1,
         all/0,
         at/1,
         elem/1,
         fetch/2,
         filter/1,
         get/2,
         get/3,
         get_and_update/3,
         key/1,
         key/2,
         'key!'/1,
         pop/2]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.Access';
'__info__'(functions) ->
    [{all,0},
     {at,1},
     {elem,1},
     {fetch,2},
     {filter,1},
     {get,2},
     {get,3},
     {get_and_update,3},
     {key,1},
     {key,2},
     {'key!',1},
     {pop,2}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.Access', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.Access', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.Access', Key);
'__info__'(deprecated) ->
    [].

all() ->
    fun all/3.

all(get, _data@1, _next@1) when is_list(_data@1) ->
    'Elixir.Enum':map(_data@1, _next@1);
all(get_and_update, _data@1, _next@1) when is_list(_data@1) ->
    all(_data@1, _next@1, __gets@1 = [], __updates@1 = []);
all(__op@1, _data@1, __next@1) ->
    error('Elixir.RuntimeError':exception(<<"Access.all/0 expected a li"
                                            "st, got: ",
                                            ('Elixir.Kernel':inspect(_data@1))/binary>>)).

all([_head@1|_rest@1], _next@1, _gets@1, _updates@1) ->
    case _next@1(_head@1) of
        {_get@1,_update@1} ->
            all(_rest@1,
                _next@1,
                [_get@1|_gets@1],
                [_update@1|_updates@1]);
        pop ->
            all(_rest@1, _next@1, [_head@1|_gets@1], _updates@1)
    end;
all([], __next@1, _gets@1, _updates@1) ->
    {lists:reverse(_gets@1),lists:reverse(_updates@1)}.

at(_index@1)
    when
        is_integer(_index@1)
        andalso
        _index@1 >= 0 ->
    fun(_op@1, _data@1, _next@1) ->
           at(_op@1, _data@1, _index@1, _next@1)
    end.

at(get, _data@1, _index@1, _next@1) when is_list(_data@1) ->
    _next@1('Elixir.Enum':at(_data@1, _index@1));
at(get_and_update, _data@1, _index@1, _next@1) when is_list(_data@1) ->
    get_and_update_at(_data@1, _index@1, _next@1, []);
at(__op@1, _data@1, __index@1, __next@1) ->
    error('Elixir.RuntimeError':exception(<<"Access.at/1 expected a lis"
                                            "t, got: ",
                                            ('Elixir.Kernel':inspect(_data@1))/binary>>)).

elem(_index@1)
    when
        is_integer(_index@1)
        andalso
        _index@1 >= 0 ->
    _pos@1 = _index@1 + 1,
    fun(get, _data@1, _next@1) when is_tuple(_data@1) ->
           _next@1(element(_pos@1, _data@1));
       (get_and_update, _data@2, _next@2) when is_tuple(_data@2) ->
           _value@1 = element(_pos@1, _data@2),
           case _next@2(_value@1) of
               {_get@1,_update@1} ->
                   {_get@1,setelement(_pos@1, _data@2, _update@1)};
               pop ->
                   error('Elixir.RuntimeError':exception(<<"cannot pop "
                                                           "data from a"
                                                           " tuple">>))
           end;
       (__op@1, _data@3, __next@1) ->
           error('Elixir.RuntimeError':exception(<<"Access.elem/1 expec"
                                                   "ted a tuple, got: ",
                                                   ('Elixir.Kernel':inspect(_data@3))/binary>>))
    end.

fetch(#{'__struct__' := __@1 = _module@1} = _container@1, _key@1)
    when is_atom(__@1) ->
    try
        _module@1:fetch(_container@1, _key@1)
    catch
        error:__@4:___STACKTRACE__@1 when __@4 == undef ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,fetch,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1);
        error:#{'__struct__' := __@5,'__exception__' := true} = __@4:___STACKTRACE__@1
            when __@5 == 'Elixir.UndefinedFunctionError' ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,fetch,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1)
    end;
fetch(_map@1, _key@1) when is_map(_map@1) ->
    case _map@1 of
        #{_key@1 := _value@1} ->
            {ok,_value@1};
        _ ->
            error
    end;
fetch(_list@1, _key@1)
    when
        is_list(_list@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _list@1) of
        {_,_value@1} ->
            {ok,_value@1};
        false ->
            error
    end;
fetch(_list@1, _key@1) when is_list(_list@1) ->
    error('Elixir.ArgumentError':exception(<<"the Access calls for keyw"
                                             "ords expect the key to be"
                                             " an atom, got: ",
                                             ('Elixir.Kernel':inspect(_key@1))/binary>>));
fetch(nil, __key@1) ->
    error.

filter(_func@1) when is_function(_func@1) ->
    fun(_op@1, _data@1, _next@1) ->
           filter(_op@1, _data@1, _func@1, _next@1)
    end.

filter(get, _data@1, _func@1, _next@1) when is_list(_data@1) ->
    'Elixir.Enum':map('Elixir.Enum':filter(_data@1, _func@1), _next@1);
filter(get_and_update, _data@1, _func@1, _next@1) when is_list(_data@1) ->
    get_and_update_filter(_data@1, _func@1, _next@1, [], []);
filter(__op@1, _data@1, __func@1, __next@1) ->
    error('Elixir.RuntimeError':exception(<<"Access.filter/1 expected a"
                                            " list, got: ",
                                            ('Elixir.Kernel':inspect(_data@1))/binary>>)).

get(__@1, __@2) ->
    get(__@1, __@2, nil).

get(#{'__struct__' := __@1 = _module@1} = _container@1,
    _key@1,
    _default@1)
    when is_atom(__@1) ->
    try _module@1:fetch(_container@1, _key@1) of
        {ok,_value@1} ->
            _value@1;
        error ->
            _default@1
    catch
        error:__@4:___STACKTRACE__@1 when __@4 == undef ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,fetch,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1);
        error:#{'__struct__' := __@5,'__exception__' := true} = __@4:___STACKTRACE__@1
            when __@5 == 'Elixir.UndefinedFunctionError' ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,fetch,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1)
    end;
get(_map@1, _key@1, _default@1) when is_map(_map@1) ->
    case _map@1 of
        #{_key@1 := _value@1} ->
            _value@1;
        _ ->
            _default@1
    end;
get(_list@1, _key@1, _default@1)
    when
        is_list(_list@1)
        andalso
        is_atom(_key@1) ->
    case lists:keyfind(_key@1, 1, _list@1) of
        {_,_value@1} ->
            _value@1;
        false ->
            _default@1
    end;
get(_list@1, _key@1, __default@1) when is_list(_list@1) ->
    error('Elixir.ArgumentError':exception(<<"the Access calls for keyw"
                                             "ords expect the key to be"
                                             " an atom, got: ",
                                             ('Elixir.Kernel':inspect(_key@1))/binary>>));
get(nil, __key@1, _default@1) ->
    _default@1.

get_and_update(#{'__struct__' := __@1 = _module@1} = _container@1,
               _key@1,
               _fun@1)
    when is_atom(__@1) ->
    try
        _module@1:get_and_update(_container@1, _key@1, _fun@1)
    catch
        error:__@4:___STACKTRACE__@1 when __@4 == undef ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,
                      get_and_update,
                      [_container@1,_key@1,_fun@1],
                      _}|
                     _] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1);
        error:#{'__struct__' := __@5,'__exception__' := true} = __@4:___STACKTRACE__@1
            when __@5 == 'Elixir.UndefinedFunctionError' ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,
                      get_and_update,
                      [_container@1,_key@1,_fun@1],
                      _}|
                     _] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1)
    end;
get_and_update(_map@1, _key@1, _fun@1) when is_map(_map@1) ->
    'Elixir.Map':get_and_update(_map@1, _key@1, _fun@1);
get_and_update(_list@1, _key@1, _fun@1) when is_list(_list@1) ->
    'Elixir.Keyword':get_and_update(_list@1, _key@1, _fun@1);
get_and_update(nil, _key@1, __fun@1) ->
    error('Elixir.ArgumentError':exception(<<"could not put/update key ",
                                             ('Elixir.Kernel':inspect(_key@1))/binary,
                                             " on a nil value">>)).

get_and_update_at([_head@1|_rest@1], 0, _next@1, _updates@1) ->
    case _next@1(_head@1) of
        {_get@1,_update@1} ->
            {_get@1,lists:reverse([_update@1|_updates@1], _rest@1)};
        pop ->
            {_head@1,lists:reverse(_updates@1, _rest@1)}
    end;
get_and_update_at([_head@1|_rest@1], _index@1, _next@1, _updates@1) ->
    get_and_update_at(_rest@1,
                      _index@1 - 1,
                      _next@1,
                      [_head@1|_updates@1]);
get_and_update_at([], __index@1, __next@1, _updates@1) ->
    {nil,lists:reverse(_updates@1)}.

get_and_update_filter([_head@1|_rest@1],
                      _func@1,
                      _next@1,
                      _updates@1,
                      _gets@1) ->
    case _func@1(_head@1) of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            get_and_update_filter(_rest@1,
                                  _func@1,
                                  _next@1,
                                  [_head@1|_updates@1],
                                  _gets@1);
        _ ->
            case _next@1(_head@1) of
                {_get@1,_update@1} ->
                    get_and_update_filter(_rest@1,
                                          _func@1,
                                          _next@1,
                                          [_update@1|_updates@1],
                                          [_get@1|_gets@1]);
                pop ->
                    get_and_update_filter(_rest@1,
                                          _func@1,
                                          _next@1,
                                          _updates@1,
                                          [_head@1|_gets@1])
            end
    end;
get_and_update_filter([], __func@1, __next@1, _updates@1, _gets@1) ->
    {lists:reverse(_gets@1),lists:reverse(_updates@1)}.

key(__@1) ->
    key(__@1, nil).

key(_key@1, _default@1) ->
    fun(get, _data@1, _next@1) ->
           _next@1('Elixir.Map':get(_data@1, _key@1, _default@1));
       (get_and_update, _data@2, _next@2) ->
           _value@1 = 'Elixir.Map':get(_data@2, _key@1, _default@1),
           case _next@2(_value@1) of
               {_get@1,_update@1} ->
                   {_get@1,_data@2#{_key@1 => _update@1}};
               pop ->
                   {_value@1,maps:remove(_key@1, _data@2)}
           end
    end.

'key!'(_key@1) ->
    fun(get, #{} = _data@1, _next@1) ->
           _next@1(maps:get(_key@1, _data@1));
       (get_and_update, #{} = _data@2, _next@2) ->
           _value@1 = maps:get(_key@1, _data@2),
           case _next@2(_value@1) of
               {_get@1,_update@1} ->
                   {_get@1,_data@2#{_key@1 => _update@1}};
               pop ->
                   {_value@1,maps:remove(_key@1, _data@2)}
           end;
       (__op@1, _data@3, __next@1) ->
           error('Elixir.RuntimeError':exception(<<"Access.key!/1 expec"
                                                   "ted a map/struct, g"
                                                   "ot: ",
                                                   ('Elixir.Kernel':inspect(_data@3))/binary>>))
    end.

pop(#{'__struct__' := __@1 = _module@1} = _container@1, _key@1)
    when is_atom(__@1) ->
    try
        _module@1:pop(_container@1, _key@1)
    catch
        error:__@4:___STACKTRACE__@1 when __@4 == undef ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,pop,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1);
        error:#{'__struct__' := __@5,'__exception__' := true} = __@4:___STACKTRACE__@1
            when __@5 == 'Elixir.UndefinedFunctionError' ->
            _exception@1 =
                'Elixir.Exception':normalize(error,
                                             __@4,
                                             ___STACKTRACE__@1),
            __@7 =
                case ___STACKTRACE__@1 of
                    [{_module@1,pop,[_container@1,_key@1],_}|_] ->
                        __@6 =
                            <<('Elixir.Kernel':inspect(_module@1))/binary,
                              " does not implement the Access behaviour">>,
                        _exception@1#{reason := __@6};
                    _ ->
                        _exception@1
                end,
            erlang:raise(error,
                         'Elixir.Kernel.Utils':raise(__@7),
                         ___STACKTRACE__@1)
    end;
pop(_map@1, _key@1) when is_map(_map@1) ->
    'Elixir.Map':pop(_map@1, _key@1);
pop(_list@1, _key@1) when is_list(_list@1) ->
    'Elixir.Keyword':pop(_list@1, _key@1);
pop(nil, _key@1) ->
    error('Elixir.ArgumentError':exception(<<"could not pop key ",
                                             ('Elixir.Kernel':inspect(_key@1))/binary,
                                             " on a nil value">>)).

