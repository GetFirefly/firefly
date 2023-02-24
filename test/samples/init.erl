-module(init).

-export([start/0]).

-import(erlang, [print/1]).

-spec start() -> ok | error.
start() ->
    Args = get_args(),
    ShouldGreet = case Args of
        {ok, [_, <<"true">> | _]} ->
            true;
        {ok, [_, <<"false">> | _]} ->
            false;
        _ ->
            false
    end,
    say_hello(ShouldGreet, <<"Hello, world!">>, <<"Nothing to say.">>).

-spec say_hello(true | false, binary(), binary()) -> ok | error.
say_hello(true, Greet, _Ignore) ->
    print(Greet);
say_hello(false, _Greet, Ignore) ->
    print(Ignore).

-spec get_args() -> {ok, [binary()]}.
get_args() ->
    {ok, init:get_plain_arguments()}.
