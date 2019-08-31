-module(run).

-export([run/0]).

run() ->
    Ast = 'Elixir.TemplateDemo.Ast':ast(),
    lumen_intrinsics:println(Ast),
    Ret = 'Elixir.DbMonDemo.AppSupervisor':start_link(Ast),

    %Ret = 'Elixir.DbMonDemo.BodySupervisor':start_link(Ast),
    
    lumen_intrinsics:println(Ret).

