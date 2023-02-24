-module(init).

-export([start/0]).

-import(erlang, [print/1]).

-spec start() -> ok | error.
start() ->
    case catch run() of
        done ->
            ok;
        Other ->
            print(<<"run failed: ">>),
            print(Other),
            error
    end.

run() ->
    case catch raise_err(throw, next) of
        next ->
            print(<<"throw: next">>),
            try
                raise_err(error, failed)
            catch
                error:Reason ->
                    print(<<"caught error">>),
                    print(Reason),
                    throw(done)
            after
                print(<<"entered after">>)
            end;
        _Other ->
            print(<<"failed to catch throw">>),
            error
    end.

raise_err(throw, Reason) ->
    throw(Reason);
raise_err(error, Reason) ->
    erlang:error(Reason).
