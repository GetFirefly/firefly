-module(init).

-export([start/0]).

-import(erlang, [print/1]).

start() ->
    Integer = 10,
    BigInteger = 8000000000000000,
    Float = 10.5,
    Float2 = 0.1,
    List = [1, 2, 3],
    Atoms = [ok, error, false, true],
    MixedList = [Integer, Float, Atoms],
    ImproperList = [Integer | 0],
    Tuple = {ok, []},
    %%TupleWTuple = {{}},
    TupleWFloat = {Float2},
    Map = #{list => List},
    print(<<"Integer: ", Integer/integer>>),
    print(<<"BigInteger: ">>),
    print(BigInteger),
    %%print(<<"Float: ", Float/float>>),
    print(<<"Float:">>),
    print(Float),
    print(<<"Float 2:">>),
    print(Float2),
    print(<<"Atoms:">>),
    print(Atoms),
    print(<<"MixedList:">>),
    print(MixedList),
    print(<<"Tuple:">>),
    print(Tuple),
    %%print(<<"TupleWTuple:">>),
    %%print(TupleWTuple),
    print(<<"TupleWFloat:">>),
    print(TupleWFloat),
    print(<<"Map:">>),
    print(Map),
    print("Charlist").
