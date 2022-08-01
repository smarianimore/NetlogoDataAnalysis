import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def compute_freqs(row, ag, col):
    tot = row[f" (learner {ag})-move-toward-chemical"] + row[f" (learner {ag})-random-walk"] + row[
        f" (learner {ag})-drop-chemical"]
    return row[col] * 100 / tot


def compute_mca(row, ag):
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


# plt.set_cmap("cividis")
# plt.style.use("cividis")
# plt.style.use("tableau-colorblind10")
plt.style.use("seaborn-colorblind")

# DOC load data and aggregate on episodes (mean)
df = pd.read_csv("data/RL-slimes/3actions-rew8-e995-01-21-Jul-2022.csv")
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
# plt.legend()
fig.tight_layout()

# DOC plot global action distribution over time
fig = plt.figure(figsize=(10, 4), dpi=200)
plt.grid()
plt.title(f"Actions per episode (50 agents, 500 steps per episode)")
plt.ylabel(f"# agents choosing action")
plt.xlabel(f"episodes")
# plt.plot(df.index, df[" move-toward-chemical"], label="move-toward-chemical", marker='o',
#          markersize=1, linewidth=0.5)
# plt.plot(df.index, df[" random-walk"], label="random-walk", marker='x', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" drop-chemical"], label="drop-chemical", marker='s', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" move-and-drop"], label="move-and-drop", marker='*', markersize=1,
#          linewidth=0.5)
# plt.plot(df.index, df[" walk-and-drop"], label="walk-and-drop", marker='^', markersize=1,
#          linewidth=0.5)
plt.plot(df.index, df[" random-walk"], label="random-walk", marker='x', markersize=1,
         linewidth=0.5)
plt.plot(df.index, df[" stand-still"], label="stand.still", marker='8', markersize=1,
         linewidth=0.5)
plt.legend()
fig.tight_layout()

# NB superseded by following routines
# DOC plot action distribution per agent (sampled randomly)
for agent in np.random.randint(0, 50, 1):  # DOC last param is # of agents to sample
    fig = plt.figure(figsize=(10, 4), dpi=200)
    plt.grid()
    plt.title(f"Actions per episode for agent {agent}")
    plt.ylabel(f"# times action chosen")
    plt.xlabel(f"episodes")
    # plt.plot(df_episode_group.index, df_episode_group[f" (learner {agent})-move-toward-chemical"], label="move-toward-chemical",
    #          marker='o', markersize=2, linewidth=1.0)
    # plt.plot(df_episode_group.index, df_episode_group[f" (learner {agent})-random-walk"], label="random-walk", marker='x', markersize=2,
    #          linewidth=1.0)
    # plt.plot(df_episode_group.index, df_episode_group[f" (learner {agent})-drop-chemical"], label="drop-chemical", marker='s',
    #          markersize=2, linewidth=1.0)
    # .stackplot could be nice but needs all data at once
    plt.scatter(df.index, df[f" (learner {agent})-move-toward-chemical"],
                label="move-toward-chemical", marker='o', s=2)
    plt.scatter(df.index, df[f" (learner {agent})-random-walk"], label="random-walk",
                marker='x', s=2)
    plt.scatter(df.index, df[f" (learner {agent})-drop-chemical"], label="drop-chemical",
                marker='s', s=2)
    plt.legend()
    fig.tight_layout()

# DOC compute action frequency per agent (and add to DF)
for agent in range(50):
    df[f" (learner {agent})-move-toward-chemical-%"] = df.apply(compute_freqs, args=(
        agent, f" (learner {agent})-move-toward-chemical"), axis=1)
    df[f" (learner {agent})-random-walk-%"] = df.apply(compute_freqs, args=(
        agent, f" (learner {agent})-random-walk"), axis=1)
    df[f" (learner {agent})-drop-chemical-%"] = df.apply(compute_freqs, args=(
        agent, f" (learner {agent})-drop-chemical"), axis=1)

for agent in range(50):
    df[f" (learner {agent}) MCA"] = df.apply(compute_mca, args=(agent,), axis=1)

# TODO "cluster" agents based on most selected action in last T episodes
marker_mca = {
    "drop-chemical": {
        'marker': 'o',
        'color': '#648FFF'},
    "random-walk": {
        'marker': 'x',
        'color': '#DC267F'},
    "move-toward-chemical": {
        'marker': '^',
        'color': '#FFB000'}
}
fig = plt.figure(figsize=(10, 4), dpi=200)
# ax = fig.add_subplot(projection='3d')
plt.grid()
plt.title(f"Agent's most chosen action per episode")
plt.ylabel(f"agents' ids")
plt.xlabel(f"episodes")
for row in df.index[:]:
    # print(row, end=' ')
    for agent in range(45, 50):
        #jitter = np.random.uniform(-0.25, 0.25)
        jitter = 0
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
custom_lines = [Line2D([], [], color='#648FFF', marker="o"),
                Line2D([], [], color='#DC267F', marker="x"),
                Line2D([], [], color='#FFB000', marker="^")]
plt.legend(custom_lines, ["drop-chemical", "random-walk", "move-toward-chemical"], loc=1, fontsize="x-small", markerscale=0.5)
fig.tight_layout()
