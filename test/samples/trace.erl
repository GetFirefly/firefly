-module(init).
-export([start/0]).
-import(erlang, [display/1]).

start() ->
    try raise_err(ohno)
    catch
        error:Reason:Trace ->
            display(Reason),
            display(Trace)
    end.


raise_err(Reason) ->
    erlang:error(Reason). 
