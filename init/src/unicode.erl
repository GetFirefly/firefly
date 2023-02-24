-module(unicode).

-export([characters_to_list/1, characters_to_list/2]).
-nifs([characters_to_list/2]).

-export_type([chardata/0, charlist/0, encoding/0, external_chardata/0,
              external_charlist/0, latin1_char/0, latin1_chardata/0,
              latin1_charlist/0, latin1_binary/0, unicode_binary/0]).

-type encoding()  :: 'latin1' | 'unicode' | 'utf8'
                   | 'utf16' | {'utf16', endian()}
                   | 'utf32' | {'utf32', endian()}.
-type endian()    :: 'big' | 'little'.
-type unicode_binary() :: binary().
-type charlist() ::
        maybe_improper_list(char() | unicode_binary() | charlist(),
                            unicode_binary() | nil()).
-type chardata() :: charlist() | unicode_binary().
-type external_unicode_binary() :: binary().
-type external_chardata() :: external_charlist() | external_unicode_binary().
-type external_charlist() ::
        maybe_improper_list(char() |
                              external_unicode_binary() |
                              external_charlist(),
                            external_unicode_binary() | nil()).
-type latin1_binary() :: binary().
-type latin1_char() :: byte().
-type latin1_chardata() :: latin1_charlist() | latin1_binary().
-type latin1_charlist() ::
        maybe_improper_list(latin1_char() |
                              latin1_binary() |
                              latin1_charlist(),
                            latin1_binary() | nil()).


-spec characters_to_list(Data,  InEncoding) -> Result when
      Data :: latin1_chardata() | chardata() | external_chardata(),
      InEncoding :: encoding(),
      Result :: list()
              | {error, list(), RestData}
              | {incomplete, list(), binary()},
      RestData :: latin1_chardata() | chardata() | external_chardata().

characters_to_list(_, _) ->
    erlang:nif_error(undef).

-spec characters_to_list(Data) -> Result when
      Data :: latin1_chardata() | chardata() | external_chardata(),
      Result :: list()
              | {error, list(), RestData}
              | {incomplete, list(), binary()},
      RestData :: latin1_chardata() | chardata() | external_chardata().

characters_to_list(ML) ->
    try
        unicode:characters_to_list(ML, unicode)
    catch
        error:Reason ->
            error_with_info(Reason, [ML])
    end.

%% We must inline these functions so that the stacktrace points to
%% the correct function.
-compile({inline, [error_with_info/2]}).

error_with_info(Reason, Args) ->
    erlang:error(Reason, Args, [{error_info, #{module => erl_stdlib_errors}}]).
