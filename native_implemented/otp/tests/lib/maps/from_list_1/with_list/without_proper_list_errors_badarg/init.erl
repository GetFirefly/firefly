-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
  try maps:from_list([hd | tl]) of
    Map -> display({map, Map})
  catch
    Class:Exception -> display({caught, Class, Exception})
  end.
