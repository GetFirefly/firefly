-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    List = [ok, true, false],
    Test = #{list => List, bin => <<"string">>},
    decompose(Test).

decompose(#{list => List} = Map) ->
    decompose(List),
    decompose_next(Map);

decompose([_|_] = List) ->
    print(List).

decompose_next(#{bin => Bin}) ->
    decompose_binary(Bin).

decompose_binary(<<>>) ->
    print("Done");
decompose_binary(<<C:8, Rest/binary>>) ->
    print(<<C:8>>),
    decompose_binary(Rest).
