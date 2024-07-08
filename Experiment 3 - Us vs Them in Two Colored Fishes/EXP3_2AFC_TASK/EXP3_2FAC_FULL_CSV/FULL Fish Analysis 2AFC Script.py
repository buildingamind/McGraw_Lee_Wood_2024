import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats

def one_sample_ttest(sample_data, null_hypothesis_mean=0.0, significance_level=0.01):
    # Calculate the T-test for the mean of ONE group of scores
    t_statistic, p_value = stats.ttest_1samp(sample_data, null_hypothesis_mean)

    # Print results
    print(f"T-statistic: {t_statistic:.10f}")
    print(f"P-value: {p_value:.10f}")

    # Determine whether to reject the null hypothesis
    if p_value < significance_level:
        print(f"With a significance level of {significance_level}, we *REJECT* the null hypothesis.")
    else:
        print(f"With a significance level of {significance_level}, we *FAIL* to reject the null hypothesis.")

    return t_statistic, p_value

DATA_PATH = fr"G:/McGraw 2024 Source/Experiment 3 - Us vs Them in Two Colored Fishes/EXP3_2AFC_TASK\EXP3_2FAC_FULL_CSV"

models = {
    "CUR": None,
    "RND": None,
    "CRF": None,
    "CTR": None,
}

# Initialize File Structure
for model_name, model_data in models.items():
    models[model_name] = {
        f"Orange1_2AFC_{model_name}.csv": None,
        f"Orange2_2AFC_{model_name}.csv": None,
        f"Orange3_2AFC_{model_name}.csv": None,
        f"Orange4_2AFC_{model_name}.csv": None,
        f"Blue1_2AFC_{model_name}.csv": None,
        f"Blue2_2AFC_{model_name}.csv": None,
        f"Blue3_2AFC_{model_name}.csv": None,
        f"Blue4_2AFC_{model_name}.csv": None,
    }

def calculate_distance(agent_x, agent_z, shoal_x, shoal_z):
    x_diff = agent_x - shoal_x
    z_diff = agent_z - shoal_z
    return np.sqrt(x_diff**2 + z_diff**2)

def calculate_social_preference_score(df):
    # Extract the name of the agent from the column names
    agent_name = next((col.split('.')[0] for col in df.columns if "agent" in col), None)
    agent_color = agent_name.split('agent')[0]

    # If no agent is found in the column names, abort the score calculation
    if agent_name is None:
        print("No agent found in the data.")
        return

    # Initialize a preference score, this will store the score of each episode
    social_preference_scores = []

    # Loop through each episode
    for episode in df['episode'].unique():
        # Calculate the distance to the orange and blue shoals for each step
        df.loc[(df['episode'] == episode) & (df['step'] > 100), 'distance_to_orange_shoal'] = calculate_distance(
            df.loc[(df['episode'] == episode) & (df['step'] > 100), 'orange shoal.xposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), 'orange shoal.zposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), f'{agent_name}.xposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), f'{agent_name}.zposition']
        )
        df.loc[(df['episode'] == episode) & (df['step'] > 100), 'distance_to_blue_shoal'] = calculate_distance(
            df.loc[(df['episode'] == episode) & (df['step'] > 100), 'blue shoal.xposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), 'blue shoal.zposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), f'{agent_name}.xposition'],
            df.loc[(df['episode'] == episode) & (df['step'] > 100), f'{agent_name}.zposition']
        )

        # Identify the shoal the agent is closest to at each time step
        # A positive result means the agent is closer to orange shoal, negative means closer to blue shoal
        nearest_shoal = df.loc[(df['episode'] == episode) & (df['step'] > 100),
                               'distance_to_blue_shoal'] > df.loc[(df['episode'] == episode) & (df['step'] > 100),
                                                                  'distance_to_orange_shoal']

        # Count the number of times the agent is closer to the same-colored shoal
        same_color_count = nearest_shoal.sum() if 'orange' in agent_color else (~nearest_shoal).sum()

        # Compute the social preference score for this episode
        social_preference_scores.append(same_color_count / len(nearest_shoal))

    # Return the average social preference score over all episodes
    return social_preference_scores, np.mean(social_preference_scores), stats.sem(social_preference_scores)


# Execute the computation for each Agent's data
for model_name, model_data in models.items():

    # Prepare an empty list to store the agents' names and their corresponding scores
    preference_scores_std = []
    agents_names = []
    preference_scores = []
    overall_results = []

    print(f"\n{model_name}:")
    for file_name, agent_data in model_data.items():
        agent_data = pd.read_csv(f"{DATA_PATH}/{file_name}").iloc[:,:-1]
        agent_name = file_name.replace("_2AFC", "").replace(".csv", "")
        scores, mean_score, score_std = calculate_social_preference_score(agent_data)
        ttest_results = one_sample_ttest(scores, null_hypothesis_mean=0.5, significance_level=0.01)

        # Describe the results of the T-Test and significance level
        print(f"2AFC Test Social Preference Score T-Test: {model_name} | {agent_name} | {ttest_results}")
        with open(rf"{DATA_PATH}/FULL_FISH_{model_name}_{agent_name}_2AFC_TTest.txt", "w") as f:
            f.write(f"2AFC Test Social Preference Score T-Test: {model_name} | {agent_name} | {ttest_results}")
            if ttest_results[1] < 0.01:
                print(f"{model_name}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
                f.write(f"{model_name}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
            else:
                print(f"{model_name}: No Significant Difference (p > 0.01) (p = {ttest_results[1]})")
                f.write(f"{model_name}: No Significant Difference (p > 0.01) (p = {ttest_results[1]}")

        print(f'Mean Social Preference Score (and Std Dev) for {model_name}: {agent_name}: ', mean_score, '(', score_std, ')')
        agents_names.append(agent_name.replace(model_name, '').replace("_", ""))
        preference_scores.append(mean_score)
        preference_scores_std.append(score_std)

        # Store results in a list of dictionaries
        for episode_number, mean_score in enumerate(scores, start=1):
            overall_results.append({
                'Agent': agent_name,
                'Episode': episode_number,
                'Social Preference Score': mean_score,
            })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(overall_results)

    # Save the DataFrame to a .csv file
    results_df.to_csv(f"{DATA_PATH}/FULL_{model_name}_2AFC_SocialPreference_by_Episode.csv", index=False)
    print(f"Results Logged to {DATA_PATH}/FULL_{model_name}_2AFC_SocialPreference_by_Episode.csv")

    # Generate the plot with error bars
    fig, ax = plt.subplots(figsize=(12, 6))

    # With this inverted social score, we can emphasize the direction of the social score more clearly
    preference_scores_inverted = [2 * (k-0.5) if i < 4 else -2 * (k-0.5) for (i, k) in enumerate(preference_scores)]

    ax.bar(agents_names, preference_scores_inverted, yerr=preference_scores_std, color=['darkorange']*4 + ['royalblue']*4,
        error_kw={'ecolor':'black', 'capsize':7, 'elinewidth': 1})
    ax.axhline(0, color='tomato', linestyle='--')
    ax.set_ylim([-1.1, 1.1])
    ax.set_title(f'Social Preference Score Across Agents ({model_name})', fontweight="bold")
    ax.set_xlabel('Agent', fontweight="bold")
    ax.set_ylabel('Social Preference Score', fontweight="bold")
    save_name = DATA_PATH + f"/2AFC_FULL_{model_name}.png"
    plt.savefig(save_name)