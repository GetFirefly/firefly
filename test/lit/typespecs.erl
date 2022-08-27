-module(init).

% With parens
-type(chars() :: [char() | any()]).
% Without parens
-type args() :: [chars() | binary()].

-spec boot(args()) -> ok | error.
boot(_Args) ->
    ok.


