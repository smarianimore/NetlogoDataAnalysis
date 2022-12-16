import os
import matplotlib.pyplot as plt
import pandas as pd
import helper

root_in = "data/RL-slimes/automation-tests/input"
root_out = "data/RL-slimes/automation-tests/output"
plots_root = "data/RL-slimes/automation-tests/plots"

whats = [" Avg reward X episode", " Avg cluster size X tick"]
whatlabels = ["Average reward", "Average cluster size"]
ylabels = ["reward", "cluster size"]

marker_mca = {  # NB reminder
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
        'color': '#FE6100'}
}

action_space = [" move-toward-chemical", " random-walk", " drop-chemical", " move-and-drop", " walk-and-drop"]
aspace_labels = ["move-toward-chemical", "random-walk", "drop-chemical", "move-and-drop", "walk-and-drop"]
#amarkers = ['o', 'x', 's']
amarkers = [marker_mca[a]['marker'] for a in aspace_labels]


def fix_fname(f):
    fname = f.rsplit('.', maxsplit=1)[0]
    fname = fname.replace(':', '_')
    fname = fname.replace('.', '_')
    fname = fname.replace(' ', '-')
    fname = fname.replace("-PM", '_PM')
    fname = fname.replace("-AM", '_AM')
    return fname


def split_datafiles(indir="data/RL-slimes/automation-tests/input", outdir="data/RL-slimes/automation-tests/output"):
    for f in os.listdir(indir):
        fp = f"{indir}/{f}"
        if os.path.isfile(fp):
            fname = fix_fname(f)
            with open(fp) as infile:
                params = []
                for _ in range(25):
                    params.append(infile.readline())
                data = infile.readlines()
            with open(f"{outdir}/{fname}.params", 'w') as outfile:
                for line in params:
                    outfile.write(line)
            with open(f"{outdir}/{fname}.csv", 'w') as outfile:
                outfile.writelines(data)


plt.style.use("seaborn-colorblind")

split_datafiles(root_in, root_out)

for f in os.listdir(root_out):
    fp = f"{root_out}/{f}"
    fext = f.rsplit('.', maxsplit=1)[1]
    if fext == "csv":
        fname = fix_fname(f)
        df = pd.read_csv(f"{fp}")
        for what, wlab, ylab in zip(whats, whatlabels, ylabels):
            fig_performance = helper.plot_what_vs_episodes(df, what, wlab, ylab)
            plt.savefig(f"{plots_root}/{fname}_{ylab}.pdf")
        fig_actions = helper.plot_actions_distribution(df, action_space, aspace_labels, amarkers)
        plt.savefig(f"{plots_root}/{fname}_actions.pdf")
