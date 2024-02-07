import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def action_freq(row, target, domain):
    """
    Computes percentage of action frequency for {agent} given in {domain} f-strings
    :param row: the row to consider
    :param target: the target action whose frequency must be computed
    :param domain: the other actions to consider (must be list of f-strings with {agent} to consider)
    :return: the percentage of {target} action frequency against actions {domain}
    """
    tot = 0
    for act in domain:
        tot += row[act]
    return row[target] * 100 / tot


def action_frequencies_df(df, agent, actions, func):
    """
    Computes actions' frequencies for {agent} by applying {func} to each action in {actions}
    :param df: the dataframe to work ok
    :param agent: the agent to consider
    :param actions: the action space to consider (must be list of f-strings with {agent} to consider)
    :param func: the function to apply ({action_freq()} mots likely)
    :return: the modified df (not a copy)
    """
    for act in actions:
        df[f"{act}-%"] = df.apply(func, args=(agent, act), axis=1)


def agent_action_freq_df(df, domain, func):
    """
    Computes actions' frequencies for each {agent} (must be in {domain} list of f-strings)
    :param df: the dataframe to work on
    :param domain: the pool of actions (column names in {df}, as f-strings with {agent} to consider in them) to consider
    :param func: the function to apply ({action_freq()} mots likely)
    :return: the modified df (not a copy)
    """
    for action in domain:
        df[f"{action}-%"] = df.apply(func, args=(action, domain))


def most_chosen_action(row, agent, domain, alable):
    """
    Computes the most chosen action amongst {domain} actions for agent {agent}
    :param row: the row to consider
    :param agent: the agent to consider
    :param domain: the action space to consider (must be list of f-strings with {agent} to consider)
    :param alable
    :return: the most chosen action amongst {domain} actions for agent {agent}
    """
    return pd.to_numeric(row[domain]).idxmax().split(f" ({alable} {agent})-")[1]


def agent_most_chosen_action_df(df, agent, domain, alable, func):
    """
    Computes the most chosen action amongst {domain} actions for each {agent} (must be in {domain} list of f-strings)
    :param df: the dataframe to work on
    :param agent: the agent to consider
    :param domain: the pool of actions (column names in {df}, as f-strings with {agent} to consider in them ) to consider
    :param alable:
    :param func: the function to apply ({most_chosen_action()} mots likely)
    :return: the modified df (not a copy)
    """
    df[f" ({alable} {agent}) MCA"] = df.apply(func, args=(agent, domain, alable), axis=1)


def plot_what_vs_episodes(df, what,
                          whatlabel, ylabel,
                          rolling=[500, 100], markers=['x', '^'], colors=['#DC267F', '#FFB000']):
    """
    Plot {what} against episodes, with rolling mean (optional)
    :param df: the dataframe to use
    :param what: column name of the feature to plot
    :param rolling: list of window size to compute rolling average (must have same len() as {markers} and {colors})
    :param whatlabel: label to put on title for {what}
    :param ylabel: label to put on y (could be same of {what})
    :param markers: list of markers associated to rolling averages (must have same len() as {rolling} and {colors})
    :param colors: list of colors associated to rolling averages (must have same len() as {rolling} and {markers})
    :return: the figure (e.g. to save on file)
    """
    fig = plt.figure(figsize=(10, 4), dpi=200)
    plt.grid()
    plt.title(f"{whatlabel} per episode")
    plt.ylabel(ylabel)
    plt.xlabel("episodes")
    plt.scatter(df.index, df[what], s=4, linewidth=1.0, color='#994F00')  #color='#648FFF')
    for window, mark, col in zip(rolling, markers, colors):
        plt.plot(df.index, df[what].rolling(window).mean(), label=f"rolling mean {window}", marker=mark,
                 markersize=.5, linewidth=.5, color=col)
    plt.legend()
    fig.tight_layout()
    return fig


def plot_actions_distribution(df, actions=[" move-toward-chemical", " random-walk", " drop-chemical"],
                              alables=["move-toward-chemical", "random-walk", "drop-chemical"],
                              markers=['o', 'x', 's']):
    """
    Plots the global actions distribution
    :param df: the dataframe to use
    :param actions: column names of the actions to plot
    :param alables: labels to assign to each action (must have same len() as {actions} and {markers})
    :param markers: marker to assign to each action (must have same len() as {actions} and {alabels})
    :return: the figure (e.g. to save on file)
    """
    fig = plt.figure(figsize=(10, 4), dpi=200)
    plt.grid()
    plt.title(f"Actions per episode")
    plt.ylabel(f"# agents choosing action")
    plt.xlabel(f"episodes")
    for act, alab, mark in zip(actions, alables, markers):
        plt.plot(df.index, df[act], label=alab, marker=mark,
                 markersize=1, linewidth=0.5)
    plt.legend()
    fig.tight_layout()
    return fig


def plot_agent_most_chosen_action(df, batch_s, batch_e, act_domain, alable, agent_mca_look={
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
        }, jit=0.35, marker_size=0.5):
    """

    :param df:
    :param batch_s:
    :param batch_e:
    :param act_domain:
    :param alable:
    :param agent_mca_look:
    :param jit:
    :param marker_size:
    :return:
    """
    custom_lines = [Line2D([], [], color=agent_mca_look[k]['color'], marker=agent_mca_look[k]['marker']) for k in act_domain]
    fig = plt.figure(figsize=(10, 4), dpi=200)
    plt.grid()
    plt.title(f"Agent's most chosen action per episode")
    plt.ylabel(f"agents' ids")
    plt.xlabel(f"episodes")
    for row in df.index[:]:
        for agent in range(batch_s, batch_e):
            jitter = np.random.uniform(-jit, jit)
            plt.plot(row, agent + jitter, marker=agent_mca_look[df.loc[row, f" ({alable} {agent}) MCA"]]['marker'],
                     markersize=marker_size, color=agent_mca_look[df.loc[row, f" ({alable} {agent}) MCA"]]['color'])
    plt.legend(custom_lines, act_domain, loc=1, fontsize="x-small", markerscale=marker_size)
    fig.tight_layout()
    return fig
