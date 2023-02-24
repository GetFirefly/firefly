%% RUN: @firefly compile --bin -o @tempfile @tests/lc.erl && @tempfile

%% CHECK: [<<"-root">>, <<"-progname">>, <<"-home">>]
-module(init).

-export([boot/1]).

boot(Args) ->
    IsValid = fun (Arg) ->
                  case Arg of
                        <<$-, _/binary>> ->
                            true;
                        _ ->
                            false
                  end
              end,
    BootArgs = [X || X <- Args, IsValid(X)],
    erlang:display(BootArgs).
