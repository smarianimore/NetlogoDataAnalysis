import os
import matplotlib.pyplot as plt
import pandas as pd
import helper

root_in = "data/RL-slimes/automation-stash/input"
root_out = "data/RL-slimes/automation-stash/output"
plots_root = "data/RL-slimes/automation-stash/plots"
header_size = 25
#header_size = 18
alable = "learner"
#alable = "turtle"

print(f"Configured for \n\t input={root_in} \n\t output={root_out} \n\t plots={plots_root}")

#whats = [" Avg reward X episode", " Avg cluster size X tick"]
#whats = [" Avg cluster size X tick"]
#whats = [" First cluster tick", " Avg cluster size X tick"]
whats = [" Avg reward X episode", " Avg cluster size X tick", " First cluster tick"]
#whats = [" First cluster tick"]

#whatlabels = ["Average reward", "Average cluster size"]
#whatlabels = ["Average cluster size"]
#whatlabels = ["Timestep when clusters appear", "Average cluster size"]
whatlabels = ["Average reward", "Average cluster size", "Timestep when clusters appear"]
#whatlabels = ["Timestep when clusters appear"]

#ylabels = ["reward", "cluster size"]
#ylabels = ["cluster size"]
#ylabels = ["episode timestep", "cluster size"]
ylabels = ["reward", "cluster size", "episode timestep"]
#ylabels = ["episode timestep"]

skip_actions = True

# (line) marker = ['o', '^', '8', 's', '*', '+', 'x']
agent_mca_look = {  # NB keys must have same name as actions' labels
    "drop-chemical": {
        'marker': 's',
        'color': '#648FFF'},
    "random-walk": {
        'marker': 'x',
        'color': '#DC267F'},
    "move-toward-chemical": {
        'marker': 'o',
        'color': '#FFB000'},
    "move-and-drop": {
        'marker': '*',
        'color': '#785EF0'},
    "walk-and-drop": {
        'marker': '^',
        'color': '#FE6100'},
    "stand-still": {
        'marker': '8',
        'color': '#000000'
    }
}

#action_space = [" move-toward-chemical", " random-walk", " drop-chemical"]
action_space = [" move-toward-chemical", " random-walk", " drop-chemical", " move-and-drop", " walk-and-drop"]
#action_space = [" random-walk", " stand-still"]
#action_space = [" move-and-drop", " walk-and-drop"]

#aspace_labels = ["move-toward-chemical", "random-walk", "drop-chemical"]
aspace_labels = ["move-toward-chemical", "random-walk", "drop-chemical", "move-and-drop", "walk-and-drop"]
#aspace_labels = ["random-walk", "stand-still"]
#aspace_labels = ["move-and-drop", "walk-and-drop"]

amarkers = [agent_mca_look[a]['marker'] for a in aspace_labels]
n_agents = 50
batch_size = 10

print(f"Action space={aspace_labels}")


def fix_fname(f):
    fname = f.rsplit('.', maxsplit=1)[0]
    fname = fname.replace(':', '_')
    fname = fname.replace('.', '_')
    fname = fname.replace(' ', '-')
    fname = fname.replace("-PM", '_PM')
    fname = fname.replace("-AM", '_AM')
    return fname


def split_datafiles(indir="data/RL-slimes/automation-stash/input", outdir="data/RL-slimes/automation-stash/output", header_size=25):
    files = os.listdir(indir)
    n_files = len(files)
    filestring = '\n\t\t '.join(files)
    print(f"\tinput files (#{n_files}): \n\t\t {filestring} \n\t\t .")
    i = 1
    for f in files:
        if ".DS_Store" in f:
            continue
        fp = f"{indir}/{f}"
        if os.path.isfile(fp):
            fname = fix_fname(f)
            print(f"\t> {fname} ...({i}/{n_files})")
            i += 1
            with open(fp) as infile:
                params = []
                for _ in range(header_size):  # NB number of lines composing .params
                    params.append(infile.readline())
                data = infile.readlines()
            out_pname = f"{fname}.params"
            with open(f"{outdir}/{out_pname}", 'w') as outfile:
                print(f"\t\t> {out_pname} ...")
                for line in params:
                    outfile.write(line)
                print("\t\t... done.")
            out_dname = f"{fname}.csv"
            with open(f"{outdir}/{out_dname}", 'w') as outfile:
                print(f"\t\t> {out_dname} ...")
                outfile.writelines(data)
                print("\t\t... done.")
            print("\t... done.")


plt.style.use("seaborn-colorblind")

print("Processing input files...")
split_datafiles(root_in, root_out, header_size)
print("... done.")

print("Processing output files...")
n_proc = 1
n_skip = 0
files = os.listdir(root_out)
n_files = int(len(files)/2)
for f in files:
    fp = f"{root_out}/{f}"
    fext = f.rsplit('.', maxsplit=1)[1]
    if fext == "csv":
        fname = fix_fname(f)
        print(f"\t> {fname}.{fext} ...({n_proc}/{n_files})")
        n_proc += 1
        df = pd.read_csv(f"{fp}")
        df.replace({" First cluster tick": -1}, 500, inplace=True)  # NB hard-code fix -1 => 500 for episodes with no clustering
        for what, wlab, ylab in zip(whats, whatlabels, ylabels):
            print(f"\t\t> Plotting {what} ...")
            plot_path = f"{plots_root}/{fname}_{ylab}.pdf"
            if os.path.exists(plot_path):
                print("\t\t... skip (already there).")
                n_skip += 1
            else:
                fig_performance = helper.plot_what_vs_episodes(df, what, wlab, ylab)
                plt.savefig(plot_path)
                plt.clf()
                plt.close(fig_performance)
                print("\t\t... done.")
        if skip_actions:
            print(f"\t\t> Asked to skip action plots ...")
            continue
        print(f"\t\t> Plotting global action space ...")
        plot_path = f"{plots_root}/{fname}_actions.pdf"
        if os.path.exists(plot_path):
            print("\t\t... skip (already there).")
            n_skip += 1
        else:
            fig_actions = helper.plot_actions_distribution(df, action_space, aspace_labels, amarkers)
            plt.savefig(plot_path)
            plt.clf()
            plt.close(fig_actions)
            print("\t\t... done.")
        print(f"\t\t> Computing agents' most chosen action ...")
        all_there = True
        for b in range(int(n_agents / batch_size)):
            plot_path = f"{plots_root}/{fname}_actions_{b+1}.pdf"
            if not os.path.exists(plot_path):
                all_there = False
                break
        if all_there:
            print("\t\t... skip (already there).")
            n_skip += int(n_agents / batch_size)
        else:
            for ag in range(n_agents):
                print(f"\t\t\t> agent {ag} ...", end=" ")
                ag_actions = [f" ({alable} {ag})-{a}" for a in aspace_labels]
                helper.agent_most_chosen_action_df(df, ag, ag_actions, alable, helper.most_chosen_action)
            print("\n\t\t... done.")
            print(f"\t\t> Plotting agents' most chosen action ...")
            for b in range(int(n_agents/batch_size)):
                print(f"\t\t\t> batch {b*batch_size}-{(b+1)*batch_size-1} ...", end=" ")
                plot_path = f"{plots_root}/{fname}_actions_{b+1}.pdf"
                if os.path.exists(plot_path):
                    print("\t\t\t... skip (already there).")
                    n_skip += 1
                else:
                    fig_mca = helper.plot_agent_most_chosen_action(df, b*batch_size, (b+1)*batch_size, aspace_labels, alable, agent_mca_look, jit=0.35, marker_size=0.5)
                    plt.savefig(plot_path)
                    plt.clf()
                    plt.close(fig_mca)
            print("\n\t\t... done.")
print(f"... done (skipped: {n_skip} steps).")
