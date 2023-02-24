-module(foo).

-export([main/1]).

main([]) ->
    ok;
main([H|T]) ->
    case T of
        [H | T2] ->
            main(T);
        [{tag, V} | T2] ->
            print(V),
            main(T2);
        [#{key => V} = M | T2] ->
            print(V),
            print(M),
            main(T2);
        [H2 | T2] ->
            print(H2),
            main(T2);
        [] ->
            ok
    end.

print(Term) ->
    try 
        printf("~p\n", Term)
    catch
        error:Reason:Trace ->
            {error, {Reason, Trace}}
    end.

printf(Format, Term) ->
    throw({error, unimplemented}).
