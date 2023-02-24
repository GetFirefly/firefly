-module(init).

-export([boot/1]).

-spec boot(BootArgs) -> no_return() when
      BootArgs :: [binary()].
boot(BootArgs) ->
    {Start0,Flags,Args} = parse_boot_args(BootArgs),
    Start = map(fun prepare_run_args/1, Start0),
    boot(Start, Flags, Args).

boot(Start, Flags, Args) ->
    erlang:display(Start),
    erlang:display(Flags),
    erlang:display(Args),
    ok.

prepare_run_args({_, L=[]}) ->
    bs2as(L);
prepare_run_args({_, L=[_]}) ->
    bs2as(L);
prepare_run_args({s, [M,F|Args]}) ->
    [b2a(M), b2a(F) | bs2as(Args)].

b2a(Bin) when is_binary(Bin) ->
    list_to_atom(b2s(Bin));
b2a(A) when is_atom(A) ->
    A.

b2s(Bin) when is_binary(Bin) ->
    try
        unicode:characters_to_list(Bin, file:native_name_encoding())
    catch
        _:_ -> binary_to_list(Bin)
    end;
b2s(L) when is_list(L) ->
    L.

bs2as(L0) when is_list(L0) ->
    map(fun b2a/1, L0);
bs2as(L) ->
    L.

map(_F,[]) ->
    [];
map(F, [X|Rest]) ->
    [F(X) | map(F, Rest)].


parse_boot_args(Args) ->
    parse_boot_args(Args, [], [], []).

parse_boot_args([B|Bs], Ss, Fs, As) ->
    case check(B) of
	start_extra_arg ->
	    {reverse(Ss),reverse(Fs),lists:reverse(As, Bs)}; % BIF
	start_arg ->
	    {S,Rest} = get_args(Bs, []),
	    parse_boot_args(Rest, [{s, S}|Ss], Fs, As);
	start_arg2 ->
	    {S,Rest} = get_args(Bs, []),
	    parse_boot_args(Rest, [{run, S}|Ss], Fs, As);
	{flag,A} ->
	    {F,Rest} = get_args(Bs, []),
	    Fl = {A,F},
	    parse_boot_args(Rest, Ss, [Fl|Fs], As);
	arg ->
	    parse_boot_args(Bs, Ss, Fs, [B|As]);
	end_args ->
	    parse_boot_args(Bs, Ss, Fs, As)
    end;
parse_boot_args([], Start, Flags, Args) ->
    {reverse(Start),reverse(Flags),reverse(Args)}.

check(<<"-extra">>) -> start_extra_arg;
check(<<"-s">>) -> start_arg;
check(<<"-run">>) -> start_arg2;
check(<<"--">>) -> end_args;
check(<<"-",Flag/binary>>) -> {flag,b2a(Flag)};
check(_) -> arg.

get_args([B|Bs], As) ->
    case check(B) of
	start_extra_arg -> {reverse(As), [B|Bs]};
	start_arg -> {reverse(As), [B|Bs]};
	start_arg2 -> {reverse(As), [B|Bs]};
	eval_arg -> {reverse(As), [B|Bs]};
	end_args -> {reverse(As), Bs};
	{flag,_} -> {reverse(As), [B|Bs]};
	arg ->
	    get_args(Bs, [B|As])
    end;
get_args([], As) -> {reverse(As),[]}.

reverse(L) ->
    lists:reverse(L).
