-module(init).
-export([start/0]).
-import(erlang, [display/1, get_keys/1, put/2]).

start() ->
  SearchValue = search_value,
  EntryKey = key,
  EntryValue = value,
  put(EntryKey, EntryValue),
  display(get_keys(SearchValue)).

