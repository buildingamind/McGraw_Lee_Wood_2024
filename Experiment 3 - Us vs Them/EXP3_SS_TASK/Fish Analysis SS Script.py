# VERIFIED 7/2/2024 TO FUNCTION AS INTENDED ON DATA PRESENT IN DIRECTORY
# Revised 11/24/2024 

# Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import math
import itertools
import warnings
from scipy.stats import ttest_ind

DATA_PATH = "/Volumes/ESD-USB/"
OUTPUT_PATH = "/Users/samanthawood/Documents/McGraw_Lee_Wood_2024/Experiment 3 - Us vs Them/EXP3_SS_TASK"

# experiment = "32O32B"
num_agents = 4
models = ['CUR', 'RND', 'CRF', 'CTR']
conditions = ['ALONE', 'FULL', 'LOWIM']

# Plot settings
model_titles = {
    'CUR': "Intrinsic Curiosity Module",
    'RND': "Random Network Distillation",
    'CRF': "Intrinsic Curiosity Module\n(random features)",
    'CTR': "Contrastive Curiosity Learning"
}
plt_fontsize = 30
plt_titlesize = 40
plt_dotsize = 16
plt_jitter = 0.08
plt_errorwidth = 3
plt_errorcap = 12
plt_barwidth = 0.5


# Function to measure Euclidean Distance
def calc_distance(x1, z1, x2, z2):
    return np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

# Function to load data
def load_data(cond, model):
    source_csv = fr"{cond}_SS_{model}.csv"
    print(f"[INFO] Loading: {DATA_PATH + source_csv}")
    df = pd.read_csv(DATA_PATH + source_csv)
    print(df.head)
    print(df.columns)
    print(len(df.columns))
    
    return df

# Function to make a numpy array with all of the average distances for each episode for each agent
def calc_episode_dists(df):
    each_episode_distances = []
    max_episodes = df['episode'].max()

    for episode in range(1, max_episodes):
        # Extract data for only currenct episode in loop
        episode_data = df[df['episode'] == episode]

        # Pull out positions for each fish (by fish type) and position (by position type_)
        fish_type = ["orangeagent", "blueagent"]
        pos_type = ["xposition", "zposition"]
        included_data = [f"{fish}_{str(num).zfill(2)}.{pos}" for fish in fish_type for num in range(1, num_agents + 1)
                        for pos in pos_type]
        positions = episode_data[included_data].values

        each_step_values = []
        for v in positions:  # We skip the first since it's init
            coords = np.array([(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)])
            distance_matrix = cdist(coords, coords)
            distance_matrix[distance_matrix == 0] = np.nan
            each_step_values.append(distance_matrix)

        episode_mean_distances = np.nanmean(each_step_values, axis=0)
        each_episode_distances.append(episode_mean_distances)

    each_episode_distances = np.array(each_episode_distances)

    return each_episode_distances

# Function to separate data into within group and between group distances
def separate_by_fish(each_episode_distances):
    # Split the Groups by Orange-Orange, Blue-Blue, Orange-Blue, Blue-Orange
    # O-to-O
    orange_to_orange_distance = each_episode_distances[:, :4, :4]
    orange_to_orange_mean = np.nanmean(orange_to_orange_distance, axis=1)

    # O-to-B
    orange_to_blue_distance = each_episode_distances[:, :4, 4:]
    orange_to_blue_mean = np.nanmean(orange_to_blue_distance, axis=1)

    # B-to-B
    blue_to_blue_distance = each_episode_distances[:, 4:, 4:]
    blue_to_blue_mean = np.nanmean(blue_to_blue_distance, axis=1)

    # B-to-O
    blue_to_orange_distance = each_episode_distances[:, 4:, :4]
    blue_to_orange_mean = np.nanmean(blue_to_orange_distance, axis=1)

     # Concatenate orange_to_orange and blue_to_blue 
    orange_to_orange_and_blue_to_blue_mean = np.hstack((orange_to_orange_mean, blue_to_blue_mean))
     # Concatenate orange_to_blue and blue_to_orange
    orange_to_blue_and_blue_to_orange_mean = np.hstack((orange_to_blue_mean, blue_to_orange_mean))

    return orange_to_orange_and_blue_to_blue_mean, orange_to_blue_and_blue_to_orange_mean

# Function to graph the results
def plot_results(orange_to_orange_and_blue_to_blue_mean_by_fish,orange_to_blue_and_blue_to_orange_mean_by_fish):
    bars_combined = [orange_to_orange_and_blue_to_blue_mean_by_fish.mean(),orange_to_blue_and_blue_to_orange_mean_by_fish.mean()]
    error_bars_combined = [orange_to_orange_and_blue_to_blue_error_by_fish, orange_to_blue_and_blue_to_orange_error_by_fish]

    # Plot bars
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.bar([0, 1], bars_combined, plt_barwidth, color=["lightgray", "cadetblue"])

    # Plot dots
    x1_jittered = np.zeros(len(orange_to_orange_and_blue_to_blue_mean_by_fish)) + np.random.uniform(-plt_jitter, plt_jitter, len(orange_to_orange_and_blue_to_blue_mean_by_fish))
    x2_jittered = np.ones(len(orange_to_blue_and_blue_to_orange_mean_by_fish)) + np.random.uniform(-plt_jitter, plt_jitter, len(orange_to_blue_and_blue_to_orange_mean_by_fish))
    ax.plot(x1_jittered, orange_to_orange_and_blue_to_blue_mean_by_fish, 'k.', markersize=plt_dotsize)
    ax.plot(x2_jittered, orange_to_blue_and_blue_to_orange_mean_by_fish, 'k.', markersize=plt_dotsize)

    # Plot error bars
    plt.errorbar([0, 1], bars_combined, yerr=error_bars_combined, fmt='*', color='black', markersize=0, elinewidth=plt_errorwidth, capsize=plt_errorcap)

    # Plot settings
    plt.ylabel('Average distance between fish\n', fontsize=plt_fontsize)
    plt.title(f'\n  {model_titles[model]} \n', fontsize=plt_titlesize)
    plt.xticks([0, 1], ['In-Group\nDistance', 'Out-group\nDistance'], fontsize=plt_fontsize)
    plt.yticks(fontsize=plt_fontsize)
    plt.tight_layout()

    # Save plot    
    plt.savefig(f'{output_dir}/{cond}_{model}_SS_WITHIN.png')
    plt.close()
    print("[INFO] Bars Generated")



with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for cond in conditions:
        for model in models:
            # Step 1: Set the output path
            output_dir = fr"{OUTPUT_PATH}/output_{cond}"

            # Step 2: Load the data
            df = load_data(cond, model)

            # Step 3: Reformat the data to get the average distances per agent per episode
            # Average defined as the average across steps in the episode
            each_episode_distances = calc_episode_dists(df)

            ### This is only to reshape so that it can be given into an Excel format
            reshaped_episode_means = each_episode_distances.reshape(-1, 64)
            output_df = pd.DataFrame(reshaped_episode_means)
            output_df.to_csv(rf"{output_dir}/{cond}_FISH_{model}_Self_Segregation_by_Episode.csv")

            # Step 4: Compute the average distances per agent per episode, but this time
            # Average is defined as the average across the other fish in the category (within or between group)
            orange_to_orange_and_blue_to_blue_mean, orange_to_blue_and_blue_to_orange_mean = separate_by_fish(each_episode_distances)        

            # Step 5: Compute the mean ingroup and output distance for each fish. Use that to compute SE across the fish
            ### within:
            orange_to_orange_and_blue_to_blue_mean_by_fish = np.nanmean(orange_to_orange_and_blue_to_blue_mean, axis=0)
            orange_to_orange_and_blue_to_blue_error_by_fish = np.std(orange_to_orange_and_blue_to_blue_mean_by_fish, ddof=1)/ np.sqrt(orange_to_orange_and_blue_to_blue_mean_by_fish.shape[0])
            ### between:
            orange_to_blue_and_blue_to_orange_mean_by_fish = np.nanmean(orange_to_blue_and_blue_to_orange_mean, axis=0)
            orange_to_blue_and_blue_to_orange_error_by_fish = np.std(orange_to_blue_and_blue_to_orange_mean_by_fish, ddof=1)/ np.sqrt(orange_to_blue_and_blue_to_orange_mean_by_fish.shape[0])

            # Step 6: Perform t-test between orange_to_blue and combined_mean
            ttest_results = ttest_ind(orange_to_orange_and_blue_to_blue_mean_by_fish, orange_to_blue_and_blue_to_orange_mean_by_fish, equal_var=False)

            # Describe the results of the T-Test and significance level
            print(f"Within Color vs. Between Color: {ttest_results}")
            with open(rf"{output_dir}/{cond}_FISH_{model}_Self_Segregation_TTest.txt", "w") as f:
                f.write(f"Within Color vs. Between Color: {ttest_results}\n")
                if ttest_results[1] < 0.01:
                    print(f"{model}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
                    f.write(f"{model}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
                else:
                    print(f"{model}: No Significant Difference (p > 0.01) (p = {ttest_results[1]})")
                    f.write(f"{model}: No Significant Difference (p > 0.01) (p = {ttest_results[1]}")
            
            # Step 7: Plot results
            plot_results(orange_to_orange_and_blue_to_blue_mean_by_fish,orange_to_blue_and_blue_to_orange_mean_by_fish)
            
            # Step 8: Save the individual data points as CSV
            individual_data = {
                'orange_to_orange_and_blue_to_blue': orange_to_orange_and_blue_to_blue_mean_by_fish,
                'orange_to_blue_and_blue_to_orange': orange_to_blue_and_blue_to_orange_mean_by_fish,
            }
            individual_data_df = pd.DataFrame(individual_data)
            individual_data_df.to_csv(rf"{output_dir}/{cond}_FISH_{model}_Self_Segregation_Individual_Data.csv", index=False)

        print("[INFO] Finished")
