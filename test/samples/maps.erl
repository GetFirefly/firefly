-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    List = [ok, true, false],
    Empty = #{},
    decompose(Empty),
    Test = #{list => List},
    decompose(Test).

decompose(#{list => List} = Map) ->
    decompose(List);
decompose(Map) when is_map(Map) ->
    print(Map);
decompose([_|_] = List) ->
    print(List),
    decompose(done);
decompose(done) ->
    print(<<"Done">>).
