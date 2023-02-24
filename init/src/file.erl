-module(file).

-export([native_name_encoding/0]).
-nifs([native_name_encoding/0]).

-spec native_name_encoding() -> latin1 | utf8.
native_name_encoding() ->
    erlang:nif_error(undef).
