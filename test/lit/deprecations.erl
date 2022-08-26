%% RUN: @firefly compile -o @tempfile @file @cwd/deprecated_module.erl && @tempfile

%% CHECK: warning: use of deprecated module
%% CHECK: deprecated_module:display
%% CHECK: ^^^^^^^^^^^^^^^^^^^^^^^^^ this function will be deprecated eventually
-module(init).

-export([boot/1]).

boot(Args) ->
    deprecated_module:display(Args).
