%% RUN: @firefly compile -C no_default_init --bin -o @tempfile @file && @tempfile

%% CHECK: maybe_expr: disabled
%% CHECK: init
%% CHECK: "init"
%% CHECK: boot
%% CHECK: 1
%% CHECK: predefined_macros.erl"
%% CHECK: 21
-module(init).

-export([boot/1]).


boot(_Args) ->
    maybe_enabled(),
    erlang:display(?MODULE),
    erlang:display(?MODULE_STRING),
    erlang:display(?FUNCTION_NAME),
    erlang:display(?FUNCTION_ARITY),
    erlang:display(?FILE),
    erlang:display(?LINE).

-if(?FEATURE_ENABLED(maybe_expr)).
maybe_enabled() ->
   erlang:display(<<"maybe_expr: enabled">>).
-else.
maybe_enabled() ->
   erlang:display(<<"maybe_expr: disabled">>).
-endif.

