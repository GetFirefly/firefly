-module(init).

-export([start/0]).
-import(erlang, [display/1]).

start() ->
  each(fun
    (AugendNumber) when is_number(AugendNumber) -> ignore;
    (AugendTerm) -> each(fun
      (AddendNumber) when is_number(AddendNumber) -> caught(fun () ->
        display({augend, AugendTerm}),
        display({addend, AddendNumber}),
        AugendTerm + AddendNumber
      end);
      (AddendTerm) ->
        display({augend, AugendTerm}),
        display({addend, AddendTerm}),
        ignore
    end)
  end).

each(Fun) ->
  apply(Fun, [atom()]).

caught(Fun) ->
  try apply(Fun, []) of
    Return ->
      display({returned, Return})
  catch
    Class:Exception ->
      display({caught, Class, Exception})
  end.

atom() ->
  Atom = atom,
  true = is_atom(Atom),
  Atom.
