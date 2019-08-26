-file("/home/build/elixir/lib/elixir/lib/string/chars.ex", 3).

-module('Elixir.String.Chars').

-callback to_string(t()) -> 'Elixir.String':t().

-spec 'impl_for!'(term()) -> atom().

-spec impl_for(term()) -> atom() | nil.

-spec '__protocol__'(module) -> 'Elixir.String.Chars';
                    (functions) -> [{to_string, 1}, ...];
                    ('consolidated?') -> boolean();
                    (impls) ->
                        not_consolidated | {consolidated, [module()]}.

-export_type([t/0]).

-type t() :: term().

-dialyzer({nowarn_function,
           [{'__protocol__',1},{impl_for,1},{'impl_for!',1}]}).

-protocol([{fallback_to_any,false}]).

-export(['__info__'/1,
         '__protocol__'/1,
         impl_for/1,
         'impl_for!'/1,
         to_string/1]).

-spec '__info__'(attributes |
                 compile |
                 functions |
                 macros |
                 md5 |
                 module |
                 deprecated) ->
                    any().

'__info__'(module) ->
    'Elixir.String.Chars';
'__info__'(functions) ->
    [{'__protocol__',1},{impl_for,1},{'impl_for!',1},{to_string,1}];
'__info__'(macros) ->
    [];
'__info__'(Key = attributes) ->
    erlang:get_module_info('Elixir.String.Chars', Key);
'__info__'(Key = compile) ->
    erlang:get_module_info('Elixir.String.Chars', Key);
'__info__'(Key = md5) ->
    erlang:get_module_info('Elixir.String.Chars', Key);
'__info__'(deprecated) ->
    [].

'__protocol__'(module) ->
    'Elixir.String.Chars';
'__protocol__'(functions) ->
    [{to_string,1}];
'__protocol__'('consolidated?') ->
    false;
'__protocol__'(impls) ->
    not_consolidated.

impl_for(#{'__struct__' := __@2 = __@1}) when is_atom(__@2) ->
    struct_impl_for(__@1);
impl_for(__@1) when is_tuple(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Tuple')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Tuple',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Tuple':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_atom(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Atom')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Atom',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Atom':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_list(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.List')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.List',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.List':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_map(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Map')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Map',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Map':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_bitstring(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.BitString')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.BitString',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.BitString':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_integer(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Integer')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Integer',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Integer':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_float(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Float')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Float',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Float':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_function(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Function')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Function',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Function':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_pid(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.PID')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.PID',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.PID':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_port(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Port')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Port',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Port':'__impl__'(target);
        false ->
            nil
    end;
impl_for(__@1) when is_reference(__@1) ->
    case
        case
            'Elixir.Code':'ensure_compiled?'('Elixir.String.Chars.Reference')
        of
            false ->
                false;
            true ->
                erlang:function_exported('Elixir.String.Chars.Reference',
                                         '__impl__',
                                         1);
            __@2 ->
                error({badbool,'and',__@2})
        end
    of
        true ->
            'Elixir.String.Chars.Reference':'__impl__'(target);
        false ->
            nil
    end;
impl_for(_) ->
    nil.

'impl_for!'(__@1) ->
    case impl_for(__@1) of
        __@2
            when
                __@2 =:= nil
                orelse
                __@2 =:= false ->
            error('Elixir.Protocol.UndefinedError':exception([{protocol,
                                                               'Elixir.String.Chars'},
                                                              {value,
                                                               __@1}]));
        __@3 ->
            __@3
    end.

struct_impl_for(__@1) ->
    __@2 = 'Elixir.Module':concat('Elixir.String.Chars', __@1),
    case
        case 'Elixir.Code':'ensure_compiled?'(__@2) of
            false ->
                false;
            true ->
                erlang:function_exported(__@2, '__impl__', 1);
            __@3 ->
                error({badbool,'and',__@3})
        end
    of
        true ->
            __@2:'__impl__'(target);
        false ->
            nil
    end.

to_string(__@1) ->
    ('impl_for!'(__@1)):to_string(__@1).

