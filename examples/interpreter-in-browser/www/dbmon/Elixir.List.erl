-file("/home/build/elixir/lib/elixir/lib/list.ex", 1).

-module('Elixir.List').

-spec first([elem]) -> nil | elem.

-spec keyreplace([tuple()], any(), non_neg_integer(), tuple()) ->
                    [tuple()].

-spec zip([list()]) -> [tuple()].

-spec keytake([tuple()], any(), non_neg_integer()) ->
                 {tuple(), [tuple()]} | nil.

-spec foldr([elem], acc, fun((elem, acc) -> acc)) -> acc.

-spec 'starts_with?'(list(), list()) -> boolean();
                    (list(), []) -> true;
                    ([], nonempty_list()) -> false.

-spec insert_at(list(), integer(), any()) -> list().

-spec myers_difference(list(),
                       list(),
                       fun((term(), term()) -> script | nil)) ->
                          script
                          when
                              script ::
                                  [{eq | ins | del | diff, list()}].

-spec last([elem]) -> nil | elem.

-spec to_integer(elixir:charlist(), 2..36) -> integer().

-spec duplicate(elem, non_neg_integer()) -> [elem].

-spec update_at([elem], integer(), fun((elem) -> any())) -> list().

-spec to_existing_atom(elixir:charlist()) -> atom().

-spec 'keymember?'([tuple()], any(), non_neg_integer()) -> boolean().

-spec delete(list(), any()) -> list().

-spec flatten(deep_list, [elem]) -> [elem]
                 when deep_list :: [elem | deep_list].

-spec wrap(nil) -> [];
          (list) -> list when list :: maybe_improper_list();
          (term) -> [term, ...] when term :: any().

-spec keydelete([tuple()], any(), non_neg_integer()) -> [tuple()].

-spec replace_at(list(), integer(), any()) -> list().

-spec keystore([tuple()], any(), non_neg_integer(), tuple()) ->
                  [tuple(), ...].

-spec to_string(unicode:charlist()) -> 'Elixir.String':t().

-spec foldl([elem], acc, fun((elem, acc) -> acc)) -> acc.

-spec to_tuple(list()) -> tuple().

-spec to_charlist(unicode:charlist()) -> elixir:charlist().

-spec keyfind([tuple()], any(), non_neg_integer(), any()) -> any().

-spec 'improper?'(maybe_improper_list()) -> boolean().

-spec pop_at(list(), integer(), any()) -> {any(), list()}.

-spec to_float(elixir:charlist()) -> float().

-spec myers_difference(list(), list()) -> [{eq | ins | del, list()}].

-spec 'ascii_printable?'(list(), limit) -> boolean()
                            when limit :: infinity | non_neg_integer().

-spec flatten(deep_list) -> list() when deep_list :: [any() | deep_list].

-spec to_atom(elixir:charlist()) -> atom().

-spec keysort([tuple()], non_neg_integer()) -> [tuple()].

-spec to_integer(elixir:charlist()) -> integer().

-spec delete_at(list(), integer()) -> list().

-export(['__info__'/1,
         'ascii_printable?'/1,
         'ascii_printable?'/2,
         delete/2,
         delete_at/2,
         duplicate/2,
         first/1,
         flatten/1,
         flatten/2,
         foldl/3,
         foldr/3,
         'improper?'/1,
         insert_at/3,
         keydelete/3,
         keyfind/3,
         keyfind/4,
         'keymember?'/3,
         keyreplace/4,
         keysort/2,
         keystore/4,
         keytake/3,
         last/1,
         myers_difference/2,
         myers_difference/3,
         pop_at/2,
         pop_at/3,
         replace_at/3,
         'starts_with?'/2,
         to_atom/1,
         to_charlist/1,
         to_existing_atom/1,
         to_float/1,
         to_integer/1,
         to_integer/2,
         to_string/1,
         to_tuple/1,
         update_at/3,
         wrap/1,
         zip/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.List';
'__info__'(functions) ->
    [{'ascii_printable?',1},
     {'ascii_printable?',2},
     {delete,2},
     {delete_at,2},
     {duplicate,2},
     {first,1},
     {flatten,1},
     {flatten,2},
     {foldl,3},
     {foldr,3},
     {'improper?',1},
     {insert_at,3},
     {keydelete,3},
     {keyfind,3},
     {keyfind,4},
     {'keymember?',3},
     {keyreplace,4},
     {keysort,2},
     {keystore,4},
     {keytake,3},
     {last,1},
     {myers_difference,2},
     {myers_difference,3},
     {pop_at,2},
     {pop_at,3},
     {replace_at,3},
     {'starts_with?',2},
     {to_atom,1},
     {to_charlist,1},
     {to_existing_atom,1},
     {to_float,1},
     {to_integer,1},
     {to_integer,2},
     {to_string,1},
     {to_tuple,1},
     {update_at,3},
     {wrap,1},
     {zip,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.List', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.List', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.List', Key);
'__info__'(deprecated) ->
    [].

'ascii_printable?'(__@1) ->
    'ascii_printable?'(__@1, infinity).

'ascii_printable?'(_list@1, _limit@1)
    when
        is_list(_list@1)
        andalso
        (_limit@1 == infinity
         orelse
         is_integer(_limit@1)
         andalso
         _limit@1 >= 0) ->
    'ascii_printable_guarded?'(_list@1, _limit@1).

'ascii_printable_guarded?'(_, 0) ->
    true;
'ascii_printable_guarded?'([_char@1|_rest@1], _counter@1)
    when
        (is_integer(_char@1)
         andalso
         _char@1 >= 32)
        andalso
        _char@1 =< 126 ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([10|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([13|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([9|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([11|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([8|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([12|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([27|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([7|_rest@1], _counter@1) ->
    'ascii_printable_guarded?'(_rest@1, decrement(_counter@1));
'ascii_printable_guarded?'([], __counter@1) ->
    true;
'ascii_printable_guarded?'(_, __counter@1) ->
    false.

compact_reverse([], _acc@1) ->
    _acc@1;
compact_reverse([{diff,_} = _fragment@1|_rest@1], _acc@1) ->
    compact_reverse(_rest@1, [_fragment@1|_acc@1]);
compact_reverse([{_kind@1,_elem@1}|_rest@1],
                [{_kind@1,_result@1}|_acc@1]) ->
    compact_reverse(_rest@1, [{_kind@1,[_elem@1|_result@1]}|_acc@1]);
compact_reverse(_rest@1,
                [{eq,_elem@1},{ins,_elem@1},{eq,_other@1}|_acc@1]) ->
    compact_reverse(_rest@1,
                    [{ins,_elem@1},{eq,_elem@1 ++ _other@1}|_acc@1]);
compact_reverse([{_kind@1,_elem@1}|_rest@1], _acc@1) ->
    compact_reverse(_rest@1, [{_kind@1,[_elem@1]}|_acc@1]).

decrement(infinity) ->
    infinity;
decrement(_counter@1) ->
    _counter@1 - 1.

delete([_item@1|_list@1], _item@1) ->
    _list@1;
delete([_other@1|_list@1], _item@1) ->
    [_other@1|delete(_list@1, _item@1)];
delete([], __item@1) ->
    [].

delete_at(_list@1, _index@1) when is_integer(_index@1) ->
    element(2, pop_at(_list@1, _index@1)).

do_insert_at([], __index@1, _value@1) ->
    [_value@1];
do_insert_at(_list@1, 0, _value@1) ->
    [_value@1|_list@1];
do_insert_at([_head@1|_tail@1], _index@1, _value@1) ->
    [_head@1|do_insert_at(_tail@1, _index@1 - 1, _value@1)].

do_pop_at([], __index@1, _default@1, _acc@1) ->
    {_default@1,lists:reverse(_acc@1)};
do_pop_at([_head@1|_tail@1], 0, __default@1, _acc@1) ->
    {_head@1,lists:reverse(_acc@1, _tail@1)};
do_pop_at([_head@1|_tail@1], _index@1, _default@1, _acc@1) ->
    do_pop_at(_tail@1, _index@1 - 1, _default@1, [_head@1|_acc@1]).

do_replace_at([], __index@1, __value@1) ->
    [];
do_replace_at([__old@1|_rest@1], 0, _value@1) ->
    [_value@1|_rest@1];
do_replace_at([_head@1|_tail@1], _index@1, _value@1) ->
    [_head@1|do_replace_at(_tail@1, _index@1 - 1, _value@1)].

do_update_at([_value@1|_list@1], 0, _fun@1) ->
    [_fun@1(_value@1)|_list@1];
do_update_at([_head@1|_tail@1], _index@1, _fun@1) ->
    [_head@1|do_update_at(_tail@1, _index@1 - 1, _fun@1)];
do_update_at([], __index@1, __fun@1) ->
    [].

do_zip(_list@1, _acc@1) ->
    _converter@1 =
        fun(_x@1, _acc@2) ->
               do_zip_each(to_list(_x@1), _acc@2)
        end,
    case lists:mapfoldl(_converter@1, [], _list@1) of
        {_,nil} ->
            lists:reverse(_acc@1);
        {_mlist@1,_heads@1} ->
            do_zip(_mlist@1, [to_tuple(lists:reverse(_heads@1))|_acc@1])
    end.

do_zip_each(_, nil) ->
    {nil,nil};
do_zip_each([_head@1|_tail@1], _acc@1) ->
    {_tail@1,[_head@1|_acc@1]};
do_zip_each([], _) ->
    {nil,nil}.

duplicate(_elem@1, _n@1) ->
    lists:duplicate(_n@1, _elem@1).

each_diagonal(_diag@1,
              _limit@1,
              __paths@1,
              _next_paths@1,
              __diff_script@1)
    when _diag@1 > _limit@1 ->
    {next,lists:reverse(_next_paths@1)};
each_diagonal(_diag@1,
              _limit@1,
              _paths@1,
              _next_paths@1,
              _diff_script@1) ->
    {_path@1,_rest@1} =
        proceed_path(_diag@1, _limit@1, _paths@1, _diff_script@1),
    case follow_snake(_path@1) of
        {cont,_path@2} ->
            each_diagonal(_diag@1 + 2,
                          _limit@1,
                          _rest@1,
                          [_path@2|_next_paths@1],
                          _diff_script@1);
        {done,_edits@1} ->
            {done,_edits@1}
    end.

find_script(_envelope@1, _max@1, _paths@1, _diff_script@1) ->
    case
        each_diagonal(- _envelope@1,
                      _envelope@1,
                      _paths@1,
                      [],
                      _diff_script@1)
    of
        {done,_edits@1} ->
            compact_reverse(_edits@1, []);
        {next,_paths@2} ->
            find_script(_envelope@1 + 1,
                        _max@1,
                        _paths@2,
                        _diff_script@1)
    end.

first([]) ->
    nil;
first([_head@1|_]) ->
    _head@1.

flatten(_list@1) ->
    lists:flatten(_list@1).

flatten(_list@1, _tail@1) ->
    lists:flatten(_list@1, _tail@1).

foldl(_list@1, _acc@1, _fun@1)
    when
        is_list(_list@1)
        andalso
        is_function(_fun@1) ->
    lists:foldl(_fun@1, _acc@1, _list@1).

foldr(_list@1, _acc@1, _fun@1)
    when
        is_list(_list@1)
        andalso
        is_function(_fun@1) ->
    lists:foldr(_fun@1, _acc@1, _list@1).

follow_snake({_y@1,[_elem@1|_rest1@1],[_elem@1|_rest2@1],_edits@1}) ->
    follow_snake({_y@1 + 1,_rest1@1,_rest2@1,[{eq,_elem@1}|_edits@1]});
follow_snake({__y@1,[],[],_edits@1}) ->
    {done,_edits@1};
follow_snake(_path@1) ->
    {cont,_path@1}.

'improper?'(_list@1)
    when
        is_list(_list@1)
        andalso
        length(_list@1) >= 0 ->
    false;
'improper?'(_list@1) when is_list(_list@1) ->
    true.

insert_at(_list@1, _index@1, _value@1)
    when
        is_list(_list@1)
        andalso
        is_integer(_index@1) ->
    case _index@1 of
        -1 ->
            _list@1 ++ [_value@1];
        _ when _index@1 < 0 ->
            case length(_list@1) + _index@1 + 1 of
                _index@2 when _index@2 < 0 ->
                    [_value@1|_list@1];
                _index@3 ->
                    do_insert_at(_list@1, _index@3, _value@1)
            end;
        _ ->
            do_insert_at(_list@1, _index@1, _value@1)
    end.

keydelete(_list@1, _key@1, _position@1) ->
    lists:keydelete(_key@1, _position@1 + 1, _list@1).

keyfind(__@1, __@2, __@3) ->
    keyfind(__@1, __@2, __@3, nil).

keyfind(_list@1, _key@1, _position@1, _default@1) ->
    case lists:keyfind(_key@1, _position@1 + 1, _list@1) of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            _default@1;
        __@2 ->
            __@2
    end.

'keymember?'(_list@1, _key@1, _position@1) ->
    lists:keymember(_key@1, _position@1 + 1, _list@1).

keyreplace(_list@1, _key@1, _position@1, _new_tuple@1) ->
    lists:keyreplace(_key@1, _position@1 + 1, _list@1, _new_tuple@1).

keysort(_list@1, _position@1) ->
    lists:keysort(_position@1 + 1, _list@1).

keystore(_list@1, _key@1, _position@1, _new_tuple@1) ->
    lists:keystore(_key@1, _position@1 + 1, _list@1, _new_tuple@1).

keytake(_list@1, _key@1, _position@1) ->
    case lists:keytake(_key@1, _position@1 + 1, _list@1) of
        {value,_item@1,_list@2} ->
            {_item@1,_list@2};
        false ->
            nil
    end.

last([]) ->
    nil;
last([_head@1]) ->
    _head@1;
last([_|_tail@1]) ->
    last(_tail@1).

move_down({_y@1,
           [_elem1@1|_rest1@1],
           [_elem2@1|_rest2@1] = _list2@1,
           _edits@1},
          _diff_script@1)
    when _diff_script@1 /= nil ->
    _diff@1 = _diff_script@1(_elem1@1, _elem2@1),
    case _diff@1 of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            {_y@1 + 1,_rest1@1,_list2@1,[{del,_elem1@1}|_edits@1]};
        _ ->
            {_y@1 + 1,_rest1@1,_rest2@1,[{diff,_diff@1}|_edits@1]}
    end;
move_down({_y@1,[_elem@1|_rest@1],_list2@1,_edits@1}, __diff_script@1) ->
    {_y@1 + 1,_rest@1,_list2@1,[{del,_elem@1}|_edits@1]};
move_down({_y@1,[],_list2@1,_edits@1}, __diff_script@1) ->
    {_y@1 + 1,[],_list2@1,_edits@1}.

move_right({_y@1,
            [_elem1@1|_rest1@1] = _list1@1,
            [_elem2@1|_rest2@1],
            _edits@1},
           _diff_script@1)
    when _diff_script@1 /= nil ->
    _diff@1 = _diff_script@1(_elem1@1, _elem2@1),
    case _diff@1 of
        __@1
            when
                __@1 =:= nil
                orelse
                __@1 =:= false ->
            {_y@1,_list1@1,_rest2@1,[{ins,_elem2@1}|_edits@1]};
        _ ->
            {_y@1 + 1,_rest1@1,_rest2@1,[{diff,_diff@1}|_edits@1]}
    end;
move_right({_y@1,_list1@1,[_elem@1|_rest@1],_edits@1}, __diff_script@1) ->
    {_y@1,_list1@1,_rest@1,[{ins,_elem@1}|_edits@1]};
move_right({_y@1,_list1@1,[],_edits@1}, __diff_script@1) ->
    {_y@1,_list1@1,[],_edits@1}.

myers_difference(_list1@1, _list2@1)
    when
        is_list(_list1@1)
        andalso
        is_list(_list2@1) ->
    myers_difference_with_diff_script(_list1@1, _list2@1, nil).

myers_difference(_list1@1, _list2@1, _diff_script@1)
    when
        (is_list(_list1@1)
         andalso
         is_list(_list2@1))
        andalso
        is_function(_diff_script@1) ->
    myers_difference_with_diff_script(_list1@1,
                                      _list2@1,
                                      _diff_script@1).

myers_difference_with_diff_script(_list1@1, _list2@1, _diff_script@1) ->
    _path@1 = {0,_list1@1,_list2@1,[]},
    find_script(0,
                length(_list1@1) + length(_list2@1),
                [_path@1],
                _diff_script@1).

pop_at(__@1, __@2) ->
    pop_at(__@1, __@2, nil).

pop_at(_list@1, _index@1, _default@1) when is_integer(_index@1) ->
    case _index@1 < 0 of
        false ->
            do_pop_at(_list@1, _index@1, _default@1, []);
        true ->
            do_pop_at(_list@1,
                      length(_list@1) + _index@1,
                      _default@1,
                      [])
    end.

proceed_path(0, 0, [_path@1], __diff_script@1) ->
    {_path@1,[]};
proceed_path(_diag@1, _limit@1, [_path@1|_] = _paths@1, _diff_script@1)
    when _diag@1 == - _limit@1 ->
    {move_down(_path@1, _diff_script@1),_paths@1};
proceed_path(_diag@1, _limit@1, [_path@1], _diff_script@1)
    when _diag@1 == _limit@1 ->
    {move_right(_path@1, _diff_script@1),[]};
proceed_path(__diag@1,
             __limit@1,
             [_path1@1,_path2@1|_rest@1],
             _diff_script@1) ->
    case element(1, _path1@1) > element(1, _path2@1) of
        false ->
            {move_down(_path2@1, _diff_script@1),[_path2@1|_rest@1]};
        true ->
            {move_right(_path1@1, _diff_script@1),[_path2@1|_rest@1]}
    end.

replace_at(_list@1, _index@1, _value@1)
    when
        is_list(_list@1)
        andalso
        is_integer(_index@1) ->
    case _index@1 < 0 of
        false ->
            do_replace_at(_list@1, _index@1, _value@1);
        true ->
            case length(_list@1) + _index@1 of
                _index@2 when _index@2 < 0 ->
                    _list@1;
                _index@3 ->
                    do_replace_at(_list@1, _index@3, _value@1)
            end
    end.

'starts_with?'([_head@1|_tail@1], [_head@1|_prefix_tail@1]) ->
    'starts_with?'(_tail@1, _prefix_tail@1);
'starts_with?'(_list@1, []) when is_list(_list@1) ->
    true;
'starts_with?'(_list@1, [_|_]) when is_list(_list@1) ->
    false.

to_atom(_charlist@1) ->
    list_to_atom(_charlist@1).

to_charlist(_list@1) when is_list(_list@1) ->
    try unicode:characters_to_list(_list@1) of
        _result@1 when is_list(_result@1) ->
            _result@1;
        {error,_encoded@1,_rest@1} ->
            error('Elixir.UnicodeConversionError':exception([{encoded,
                                                              _encoded@1},
                                                             {rest,
                                                              _rest@1},
                                                             {kind,
                                                              invalid}]));
        {incomplete,_encoded@2,_rest@2} ->
            error('Elixir.UnicodeConversionError':exception([{encoded,
                                                              _encoded@2},
                                                             {rest,
                                                              _rest@2},
                                                             {kind,
                                                              incomplete}]))
    catch
        error:__@3
            when
                __@3 == badarg
                orelse
                tuple_size(__@3) == 2
                andalso
                element(1, __@3) == badarg ->
            error('Elixir.ArgumentError':exception(<<"cannot convert th"
                                                     "e given list to a"
                                                     " charlist.\n\nTo "
                                                     "be converted to a"
                                                     " charlist, a list"
                                                     " must contain onl"
                                                     "y:\n\n  * strings"
                                                     "\n  * integers re"
                                                     "presenting Unicod"
                                                     "e codepoints\n  *"
                                                     " or a list contai"
                                                     "ning one of these"
                                                     " three elements\n"
                                                     "\nPlease check th"
                                                     "e given list or c"
                                                     "all inspect/1 to "
                                                     "get the list repr"
                                                     "esentation, got:"
                                                     "\n\n",
                                                     ('Elixir.Kernel':inspect(_list@1))/binary,
                                                     "\n">>));
        error:#{'__struct__' := __@4,'__exception__' := true} = __@3
            when __@4 == 'Elixir.ArgumentError' ->
            error('Elixir.ArgumentError':exception(<<"cannot convert th"
                                                     "e given list to a"
                                                     " charlist.\n\nTo "
                                                     "be converted to a"
                                                     " charlist, a list"
                                                     " must contain onl"
                                                     "y:\n\n  * strings"
                                                     "\n  * integers re"
                                                     "presenting Unicod"
                                                     "e codepoints\n  *"
                                                     " or a list contai"
                                                     "ning one of these"
                                                     " three elements\n"
                                                     "\nPlease check th"
                                                     "e given list or c"
                                                     "all inspect/1 to "
                                                     "get the list repr"
                                                     "esentation, got:"
                                                     "\n\n",
                                                     ('Elixir.Kernel':inspect(_list@1))/binary,
                                                     "\n">>))
    end.

to_existing_atom(_charlist@1) ->
    list_to_existing_atom(_charlist@1).

to_float(_charlist@1) ->
    list_to_float(_charlist@1).

to_integer(_charlist@1) ->
    list_to_integer(_charlist@1).

to_integer(_charlist@1, _base@1) ->
    list_to_integer(_charlist@1, _base@1).

to_list(_tuple@1) when is_tuple(_tuple@1) ->
    tuple_to_list(_tuple@1);
to_list(_list@1) when is_list(_list@1) ->
    _list@1.

to_string(_list@1) when is_list(_list@1) ->
    try unicode:characters_to_binary(_list@1) of
        _result@1 when is_binary(_result@1) ->
            _result@1;
        {error,_encoded@1,_rest@1} ->
            error('Elixir.UnicodeConversionError':exception([{encoded,
                                                              _encoded@1},
                                                             {rest,
                                                              _rest@1},
                                                             {kind,
                                                              invalid}]));
        {incomplete,_encoded@2,_rest@2} ->
            error('Elixir.UnicodeConversionError':exception([{encoded,
                                                              _encoded@2},
                                                             {rest,
                                                              _rest@2},
                                                             {kind,
                                                              incomplete}]))
    catch
        error:__@3
            when
                __@3 == badarg
                orelse
                tuple_size(__@3) == 2
                andalso
                element(1, __@3) == badarg ->
            error('Elixir.ArgumentError':exception(<<"cannot convert th"
                                                     "e given list to a"
                                                     " string.\n\nTo be"
                                                     " converted to a s"
                                                     "tring, a list mus"
                                                     "t contain only:\n"
                                                     "\n  * strings\n  "
                                                     "* integers repres"
                                                     "enting Unicode co"
                                                     "depoints\n  * or "
                                                     "a list containing"
                                                     " one of these thr"
                                                     "ee elements\n\nPl"
                                                     "ease check the gi"
                                                     "ven list or call "
                                                     "inspect/1 to get "
                                                     "the list represen"
                                                     "tation, got:\n\n",
                                                     ('Elixir.Kernel':inspect(_list@1))/binary,
                                                     "\n">>));
        error:#{'__struct__' := __@4,'__exception__' := true} = __@3
            when __@4 == 'Elixir.ArgumentError' ->
            error('Elixir.ArgumentError':exception(<<"cannot convert th"
                                                     "e given list to a"
                                                     " string.\n\nTo be"
                                                     " converted to a s"
                                                     "tring, a list mus"
                                                     "t contain only:\n"
                                                     "\n  * strings\n  "
                                                     "* integers repres"
                                                     "enting Unicode co"
                                                     "depoints\n  * or "
                                                     "a list containing"
                                                     " one of these thr"
                                                     "ee elements\n\nPl"
                                                     "ease check the gi"
                                                     "ven list or call "
                                                     "inspect/1 to get "
                                                     "the list represen"
                                                     "tation, got:\n\n",
                                                     ('Elixir.Kernel':inspect(_list@1))/binary,
                                                     "\n">>))
    end.

to_tuple(_list@1) ->
    list_to_tuple(_list@1).

update_at(_list@1, _index@1, _fun@1)
    when
        (is_list(_list@1)
         andalso
         is_function(_fun@1))
        andalso
        is_integer(_index@1) ->
    case _index@1 < 0 of
        false ->
            do_update_at(_list@1, _index@1, _fun@1);
        true ->
            case length(_list@1) + _index@1 of
                _index@2 when _index@2 < 0 ->
                    _list@1;
                _index@3 ->
                    do_update_at(_list@1, _index@3, _fun@1)
            end
    end.

wrap(_list@1) when is_list(_list@1) ->
    _list@1;
wrap(nil) ->
    [];
wrap(_other@1) ->
    [_other@1].

zip([]) ->
    [];
zip(_list_of_lists@1) when is_list(_list_of_lists@1) ->
    do_zip(_list_of_lists@1, []).

