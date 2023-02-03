import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def compute_freqs_3(row, ag, col):
    tot = row[f" (learner {ag})-move-toward-chemical"] + row[f" (learner {ag})-random-walk"] + row[
        f" (learner {ag})-drop-chemical"]
    return row[col] * 100 / tot


def compute_freqs_5(row, ag, col):
    tot = row[f" (learner {ag})-move-toward-chemical"] + row[f" (learner {ag})-random-walk"] + row[
        f" (learner {ag})-drop-chemical"] + row[f" (learner {ag})-move-and-drop"] + row[f" (learner {ag})-walk-and-drop"]
    return row[col] * 100 / tot


def compute_mca_3(row, ag):
    if row[f" (learner {ag})-move-toward-chemical-%"] > row[f" (learner {ag})-random-walk-%"]:
        if row[f" (learner {ag})-move-toward-chemical-%"] > row[f" (learner {ag})-drop-chemical-%"]:
            mca = "move-toward-chemical"
        else:
            mca = "drop-chemical"
    else:
        if row[f" (learner {ag})-random-walk-%"] > row[f" (learner {ag})-drop-chemical-%"]:
            mca = "random-walk"
        else:
            mca = "drop-chemical"
    return mca


def compute_mca_5(row, ag):
    return pd.to_numeric(row[[f" (learner {ag})-move-toward-chemical",
                f" (learner {ag})-random-walk",
                f" (learner {ag})-drop-chemical",
                f" (learner {ag})-move-and-drop",
                f" (learner {ag})-walk-and-drop"]]).idxmax().split(f" (learner {ag})-")[1]


# plt.set_cmap("cividis")
# plt.style.use("cividis")
# plt.style.use("tableau-colorblind10")
plt.style.use("seaborn-colorblind")


# DOC load data and aggregate on episodes (mean)
df = pd.read_csv("data/RL-slimes/behavioural/5actions/BS-test-5actions-12_58_34_343_AM-18-Oct-2022.csv")
print(df.head())
# NB no longer useful for new data (see episode datapoints in .csv file!)
df = df.groupby(["Episode"]).mean()
print(df.head())
# (line) marker = ['o', '^', '8', 's', '*', '+', 'x']
# (bar) hatch = * + - . / O X \ o x |


# DOC plot average reward over time
fig = plt.figure(figsize=(10, 4), dpi=200)
plt.grid()
plt.title(f"Average reward per episode (50 agents, 500 steps per episode)")
plt.ylabel(f"reward")
plt.xlabel(f"episodes")
plt.scatter(df.index, df[" Avg reward X episode"], s=4, linewidth=1.0, color='#648FFF')
plt.plot(df.index, df[" Avg reward X episode"].rolling(500).mean(), label="rolling mean 500", marker='x',
         markersize=.5, linewidth=.5, color='#DC267F')
plt.plot(df.index, df[" Avg reward X episode"].rolling(100).mean(), label="rolling mean 100", marker='^',
         markersize=.5, linewidth=.5, color='#FFB000')
# plt.plot(df.index, df[" Avg reward X episode"].rolling(5).mean(), label="rolling mean 5", marker='x',
#          markersize=.5, linewidth=.5, color='#DC267F')
# plt.plot(df.index, df[" Avg reward X episode"].rolling(10).mean(), label="rolling mean 10", marker='^',
#          markersize=.5, linewidth=.5, color='#FFB000')
plt.legend()
fig.tight_layout()

# DOC plot cluster size over time
fig = plt.figure(figsize=(10, 4), dpi=200)
plt.grid()
plt.title(f"Average cluster size per episode (50 agents, 500 steps per episode)")
plt.ylabel(f"# agents within cluster radius (10 patches)")
plt.xlabel(f"episodes")
plt.scatter(df.loc[df[" Avg cluster size X tick"] != 0].index, df[" Avg cluster size X tick"].loc[df[" Avg cluster size X tick"] != 0], s=8, linewidth=1.0, color='#648FFF')
# plt.plot(df.loc[df[" Avg cluster size X tick"] != 0].index, df[" Avg cluster size X tick"].loc[df[" Avg cluster size X tick"] != 0].rolling(500).mean(), label="rolling mean 500", marker='x',
#          markersize=.5, linewidth=.5, color='#DC267F')
# plt.plot(df.loc[df[" Avg cluster size X tick"] != 0].index, df[" Avg cluster size X tick"].loc[df[" Avg cluster size X tick"] != 0].rolling(100).mean(), label="rolling mean 100", marker='^',
#          markersize=.5, linewidth=.5, color='#FFB000')
plt.plot(df.index, df[" Avg cluster size X tick"].rolling(500).mean(), label="rolling mean 500", marker='x',
         markersize=.5, linewidth=.5, color='#DC267F')
plt.plot(df.index, df[" Avg cluster size X tick"].rolling(100).mean(), label="rolling mean 100", marker='^',
         markersize=.5, linewidth=.5, color='#FFB000')
plt.legend()
fig.tight_layout()

# DOC plot global action distribution over time
fig = plt.figure(figsize=(10, 4), dpi=200)
plt.grid()
plt.title(f"Actions per episode (50 agents, 500 steps per episode)")
plt.ylabel(f"# agents choosing action")
plt.xlabel(f"episodes")
plt.plot(df.index, df[" move-toward-chemical"], label="move-toward-chemical", marker='o',
         markersize=1, linewidth=0.5)
plt.plot(df.index, df[" random-walk"], label="random-walk", marker='x', markersize=1,
         linewidth=0.5)
plt.plot(df.index, df[" drop-chemical"], label="drop-chemical", marker='s', markersize=1,
         linewidth=0.5)
# plt.plot(df.index, df[" move-and-drop"], label="move-and-drop", marker='*', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" walk-and-drop"], label="walk-and-drop", marker='^', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" random-walk"], label="random-walk", marker='x', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" stand-still"], label="stand-still", marker='8', markersize=1,
#          linewidth=0.5)
plt.legend()
fig.tight_layout()

# DOC compute action frequency per agent (and add to DF)
for agent in range(50):
    df[f" (learner {agent})-move-toward-chemical-%"] = df.apply(compute_freqs_5, args=(
        agent, f" (learner {agent})-move-toward-chemical"), axis=1)
    df[f" (learner {agent})-random-walk-%"] = df.apply(compute_freqs_5, args=(
        agent, f" (learner {agent})-random-walk"), axis=1)
    df[f" (learner {agent})-drop-chemical-%"] = df.apply(compute_freqs_5, args=(
        agent, f" (learner {agent})-drop-chemical"), axis=1)
    df[f" (learner {agent})-move-and-drop-%"] = df.apply(compute_freqs_5, args=(
        agent, f" (learner {agent})-move-and-drop"), axis=1)
    df[f" (learner {agent})-walk-and-drop-%"] = df.apply(compute_freqs_5, args=(
        agent, f" (learner {agent})-walk-and-drop"), axis=1)

for agent in range(50):
    df[f" (learner {agent}) MCA"] = df.apply(compute_mca_5, args=(agent,), axis=1)

# TODO "cluster" agents based on most selected action in last T episodes
marker_mca = {
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
fig = plt.figure(figsize=(10, 4), dpi=200)
# ax = fig.add_subplot(projection='3d')
plt.grid()
plt.title(f"Agent's most chosen action per episode")
plt.ylabel(f"agents' ids")
plt.xlabel(f"episodes")
for row in df.index[:]:
    # print(row, end=' ')
    for agent in range(40, 50):
        jitter = np.random.uniform(-0.35, 0.35)
        #jitter = 0
        # print(agent, df_episode_group.loc[row, f" (learner {agent}) MCA"], end=' ')
        plt.plot(row, agent + jitter, marker=marker_mca[df.loc[row, f" (learner {agent}) MCA"]]['marker'],
                 markersize=0.5, color=marker_mca[df.loc[row, f" (learner {agent}) MCA"]]['color'])
    # print()
    # plt.scatter(df_episode_group.index, pd.Int64Index(range(50), name="Agents' ids"), hue=df_episode_group[f" (learner {agent}) MCA"],
    #             #marker=marker_mca[df_episode_group[f" (learner {agent}) MCA"]],
    #             s=4,
    #             label=df_episode_group[f" (learner {agent}) MCA"])
    # plt.plot(df_episode_group.index, df_episode_group[f" (learner {agent}) MCA"])
    # plt.scatter(df_episode_group.index, df_episode_group[f" (learner {agent}) MCA"])
    # ax.scatter(pd.Int64Index(range(50), df_episode_group.index, df_episode_group[f" (learner {agent}) MCA"]))
custom_lines = [Line2D([], [], color='#648FFF', marker="s"),
                Line2D([], [], color='#DC267F', marker="x"),
                Line2D([], [], color='#FFB000', marker="o"),
                Line2D([], [], color='#785EF0', marker="*"),
                Line2D([], [], color='#FE6100', marker="^")]
# plt.legend(custom_lines, ["drop-chemical", "random-walk", "move-toward-chemical"],
#            loc=1, fontsize="x-small", markerscale=0.5)
plt.legend(custom_lines, ["drop-chemical", "random-walk", "move-toward-chemical", "move-and-drop", "walk-and-drop"],
           loc=1, fontsize="x-small", markerscale=0.5)
fig.tight_layout()
plt.clf()
plt.close(fig)
