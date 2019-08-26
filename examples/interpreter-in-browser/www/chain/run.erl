-module(run).

run() ->
    lumen_intrinsics:println(running_chain),
    'Elixir.Chain':dom(100).
