-module(init).
-export([start/0]).
-import(erlang, [apply/2, display/1]).

start() ->
  try apply(
        fun (A) ->
          A
        end,
        []
      ) of
    Return ->
      display({returned, Return})
  catch
    Class:Exception ->
      case Exception of
        {badarity, {_, Args}} -> display({caught, Class, badarity, with, args, Args});
        _ -> display({caught, Class, Exception})
      end
  end.
