%% RUN: @lumen compile -o @tempfile @file && @tempfile

%% CHECK: maybe_expr: disabled
%% CHECK: init
%% CHECK: "init"
%% CHECK: boot
%% CHECK: 1
%% CHECK: predefined_macros.erl"
%% CHECK: 18
-module(init).

-export([boot/1]).
-deprecated({display, 1, eventually}).
-deprecated({'_', '_', eventually}).

boot(Args) ->
    display(Args).


display(Arg) ->
    erlang:display(Arg).
