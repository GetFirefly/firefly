%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: <<"Flag: root Value:
%% CHECK: <<"Flag: progname Value: 
%% CHECK: <<"Flag: home Value:
-module(init).

-export([boot/1]).

boot([]) ->
    ok;
boot([<<$-, Flag/binary>>, Value | Rest]) ->
    erlang:display(<<"Flag: ", Flag/binary, $\s, "Value: ", Value/binary>>),
    boot(Rest);
boot([Other | Rest]) ->
    erlang:display(<<"Plain: ", Other/binary>>),
    boot(Rest).
 
