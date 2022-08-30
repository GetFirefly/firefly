%% RUN: @firefly compile -Z analyze_only @file @tests/deprecated_module.erl 2>&1

%% CHECK: use of deprecated module
%% CHECK: deprecated_module:display
%% CHECK: this function will be deprecated eventually
-module(init).

-export([boot/1]).

boot(Args) ->
    deprecated_module:display(Args).
