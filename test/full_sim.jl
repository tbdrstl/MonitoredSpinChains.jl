using MonitoredSpinChains

params = Dict(
        "name" => "test",
        "systemSize" => [6],
        "meas_steps" => [x->x^3],
        "average" => [1_000],
        "bc" => [:pbc, :obc],
        "initialState" => [flat_spin1],
        "measurement" => [true],
        "feedback" => [:Z],
        "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
        "observables" => [:OP,:EEfin, :M],
        "trajectories_averaged" => [true],
        "thermalizationSteps" => [x->0],
        "meas_every" => [x->1],
        "model" => ["motzkin"],
    )

    mkpath(params["result_folder"])
    
    sim = create_simulation(params)

    simulate(sim)


#=


f = load("test/test_data/average/1091225893709352897.jld2");
obs = f["observables"];
er = f["errors"];

tp = obs.total_proj[:,1];
tpe = er.total_proj[:,1];
xs = [10^x for x in 0:3];

begin
fig, ax = scatter(tp[1:div(6^4,2)])
errorbars!(ax, 1:div(6^4,2), tp[1:div(6^4,2)],tpe[1:div(6^4,2)], color=:black)
ax.xscale=log10; ax.yscale=log10
lines!(ax, xs, xs.^-0.33 .*0.5, color=Cycled(2))
end

fig


=#