%% RUN: @firefly compile --bin -o @tempfile @file && @tempfile

%% CHECK: custom_atom
-module(init).

-export([boot/1]).


boot(_Args) ->
    do_boot(custom_atom).

do_boot(Atom) when is_atom(Atom) ->
    erlang:display(Atom).
