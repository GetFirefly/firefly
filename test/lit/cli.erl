-module(init).

-export([boot/1]).

-import(erlang, [display/1]).

-spec boot([term()]) -> ok | error.
boot([_Arg0 | Args]) ->
  PlainArgs = get_plain_arguments(Args),
  ShouldGreet = case PlainArgs of
                  [<<"true">> | _] ->
                    true;
                  [<<"false">> | _] ->
                    false;
                  _ ->
                    false
                end,
  say_hello(ShouldGreet, <<"Hello, world!">>, <<"Nothing to say.">>).

-spec say_hello(true | false, binary(), binary()) -> ok | error.
say_hello(true, Greet, _Ignore) ->
  display(Greet);
say_hello(false, _Greet, Ignore) ->
  display(Ignore).

get_plain_arguments(Args) when is_list(Args) ->
    get_plain_arguments1(Args, []).

get_plain_arguments1([], Acc) -> lists:reverse(Acc, []);
get_plain_arguments1([<<$-, _Flag/binary>>, _Value | Rest], Acc) ->
    get_plain_arguments1(Rest, Acc);
get_plain_arguments1([Arg | Rest], Acc) ->
    get_plain_arguments1(Rest, [Arg | Acc]).
