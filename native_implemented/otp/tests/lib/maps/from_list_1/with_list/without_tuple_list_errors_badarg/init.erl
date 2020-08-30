-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  try maps:from_list([[key, value]]) of
    Map -> display({map, Map})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.
