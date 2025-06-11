using MonitoredSpinChains


params = Dict(
    "name" => "motzkin",
    "systemSize" => 6:2:14,
    "meas_steps" => [x->1],
    "average" => [1],
    "bc" => [:pbc],
    "initialState" => [flat_spin1],
    "measurement" => [true],
    "feedback" => [:Z],
    "result_folder" => joinpath(@__DIR__, "..", "test/test_data"),
    "observables" => [:OP,:EEfin],
    "trajectories_averaged" => [true],
    "thermalizationSteps" => [x->x^4],
    "meas_every" => [x->1],
    "model" => ["aklt"],
)

mkpath(params["result_folder"])

sim = create_simulation(params)

simulate(sim)


#=


f1_h1 = load("./data/aklt/FinalEntanglement/2814621704739593187.jld2");
f2_h1 = load("./data/aklt/FinalEntanglement/5431095733213027281.jld2");
f3_h1 = load("./data/aklt/FinalEntanglement/7859136725201680207.jld2");
f4_h1 = load("./data/aklt/FinalEntanglement/9717952960700774922.jld2");
f5_h1 = load("./data/aklt/FinalEntanglement/14576885274364748225.jld2");

obs1_h1 = f1_h1["observables"];
obs2_h1 = f2_h1["observables"];
obs3_h1 = f3_h1["observables"];
obs4_h1 = f4_h1["observables"];
obs5_h1 = f5_h1["observables"];

ee1_h1 = obs1_h1.entanglement_entropy;
ee2_h1 = obs2_h1.entanglement_entropy;
ee3_h1 = obs3_h1.entanglement_entropy;
ee4_h1 = obs4_h1.entanglement_entropy;
ee5_h1 = obs5_h1.entanglement_entropy;

ees_h1 = [ee1_h1,ee2_h1,ee3_h1,ee4_h1,ee5_h1]

sort!(ees_h1,by=x->length(x))

fig_ee_h1 = Figure();
ax_ee_h1 = Axis(fig_ee_h1[1,1]);

for ee in ees
    scatterlines!(ax_ee_h1, (1:length(ee_h1))./(2*length(ee_h1)),ee_h1)
end

ee_h1_end = [e[end] for e in ees_h1]
Ls_ee_h1 = 6:2:14
scatter!(ax_ee_h1, Ls, ee_end);
lines!(ax_ee_h1, Ls, log10.(Ls).*1.1.+0.45)
ax_ee_h1.xscale=log10
ax_ee_h1.xticks=6:2:14
fig_ee_h1

ax.xscale=log10#; ax.yscale=log10;

fig

=#  


#=
f = load("test/test_data/average/18057850618019368183.jld2")

obs = f["observables"];
er = f["errors"];

tp = obs.total_proj[:,1];
tp2 = obs.total_proj[:,3]
v = tp2 .- tp.^2
tpe = er.total_proj[:,1];
xs = [10^x for x in 0:3];

begin
fig, ax = scatter(tp)
# errorbars!(ax,1:length(tp), tp,tpe, color=:black)
ax.xscale=log10#; ax.yscale=log10;
# lines!(ax, xs, xs.^-0.33 .*0.5, color=Cycled(2))
end

fig

println(f["circuit"].bc)

=#