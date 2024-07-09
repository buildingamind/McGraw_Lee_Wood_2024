# VERIFIED 7/2/2024 TO FUNCTION AS INTENDED ON DATA PRESENT IN DIRECTORY

# Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import math
import itertools
import warnings
from scipy.stats import ttest_ind


# Function to measure Euclidean Distance
def calc_distance(x1, z1, x2, z2):
    return np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

# experiment = "32O32B"
num_agents = 4
models = ['CUR', 'RND', 'CRF', 'CTR']

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for model in models:
        # Load data
        DATA_PATH = "G:/McGraw 2024 Source/Experiment 3 - Us vs Them in Two Colored Fishes/EXP3_SS_TASK/EXP3_SS_FULL_CSV/"
        source_csv = fr"FULL_SS_{model}.csv"
        print(f"[INFO] Loading: {DATA_PATH + source_csv}")
        df = pd.read_csv(DATA_PATH + source_csv)
        print(df.head)
        print(df.columns)
        print(len(df.columns))

        # List to hold images for the GIF
        images = []
        distance_matrix_list = []
        max_episodes = 200

        positions = []
        for i in range(num_agents):
            positions.append([df[f'blueagent_{i + 1:02}.xposition'].mean(),
                            df[f'blueagent_{i + 1:02}.zposition'].mean()])
        for i in range(num_agents):
            positions.append([df[f'orangeagent_{i + 1:02}.xposition'].mean(),
                            df[f'orangeagent_{i + 1:02}.zposition'].mean()])
        # Calculate distances
        dist_matrix = squareform(pdist(positions))
        distance_matrix_list.append(dist_matrix)

        # Find the maximum distance.
        vmax = np.max([np.max(dm) for dm in distance_matrix_list])

        # Visualize matrix with Matplotlib
        plt.figure(figsize=(8, 8))
        
        # plt.imshow(dist_matrix, cmap='viridis', vmin=0, vmax=vmax)
        plt.imshow(dist_matrix, cmap='viridis', vmin=0, vmax=0.5)
        plt.colorbar(label='Distance')
        plt.title(f'Self-Grouping Task from Socially Reared Agents\n({model} = 1.0)  \n', fontweight='bold',
                fontsize=15)
        plt.savefig(rf'{DATA_PATH}/FULL_{model}_distance.png')
        plt.close()
        # print("[INFO] Grid Generated")

        # Extract the x and z positions for all agents
        orange_nni_data = []
        blue_nni_data = []
        mixed_nni_data = []

        each_step_distances = []
        each_episode_distances = []

        max_episodes = df['episode'].max()

        for episode in range(1, max_episodes):
            episode_data = df[df['episode'] == episode]

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

        # This is only to reshape so that it can be given into an Excel format
        reshaped_episode_means = each_episode_distances.reshape(-1, 64)

        output_df = pd.DataFrame(reshaped_episode_means)
        output_df.to_csv(rf"{DATA_PATH}/FULL_FISH_{model}_Self_Segregation_by_Episode.csv")
        ###############################################################################

        # Split the Orange Groups
        orange_to_orange_distance = each_episode_distances[:, :4, :4]
        orange_to_orange_mean = np.nanmean(orange_to_orange_distance, axis=1)

        # Split the Orange Groups
        orange_to_blue_distance = each_episode_distances[:, :4, 4:]
        orange_to_blue_mean = np.nanmean(orange_to_blue_distance, axis=1)

        blue_to_blue_distance = each_episode_distances[:, 4:, 4:]
        blue_to_blue_mean = np.nanmean(blue_to_blue_distance, axis=1)

        # Calculate Standard Error
        orange_to_orange_error = np.std(orange_to_orange_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])
        orange_to_blue_error = np.std(orange_to_blue_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])
        blue_to_blue_error = np.std(blue_to_blue_mean, axis=0) / np.sqrt(orange_to_orange_distance.shape[0])

        # Concatenate orange_to_orange and blue_to_blue then see if the mean is significnatly greater than orange_to_blue
        orange_to_orange_and_blue_to_blue_mean = np.hstack((orange_to_orange_mean, blue_to_blue_mean))
        orange_to_orange_and_blue_to_blue_error = np.hstack((orange_to_orange_error, blue_to_blue_error))
        # Find the mean distance between all orange and blue fish

        # Perform t-test between orange_to_blue and combined_mean
        ttest_results = ttest_ind(np.nanmean(orange_to_orange_and_blue_to_blue_mean, axis=0), np.nanmean(orange_to_blue_mean, axis=0), equal_var=False)

        # Describe the results of the T-Test and significance level
        print(f"Orange to Blue vs. Orange to Orange and Blue to Blue: {ttest_results}")
        with open(rf"{DATA_PATH}/FULL_FISH_{model}_Self_Segregation_TTest.txt", "w") as f:
            f.write(f"Orange to Blue vs. Orange to Orange and Blue to Blue: {ttest_results}\n")
            if ttest_results[1] < 0.01:
                print(f"{model}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
                f.write(f"{model}: *** Significant Difference (p < 0.01) (p = {ttest_results[1]})")
            else:
                print(f"{model}: No Significant Difference (p > 0.01) (p = {ttest_results[1]})")
                f.write(f"{model}: No Significant Difference (p > 0.01) (p = {ttest_results[1]}")
        
        width = 0.5
        
        # Separate
        bars_separate = [orange_to_orange_mean.mean(),
                blue_to_blue_mean.mean(),
                orange_to_blue_mean.mean()]

        error_bars_separate = [orange_to_orange_error.mean(),
                    blue_to_blue_error.mean(),
                    orange_to_blue_error.mean()]

        print(bars_separate)
        print(error_bars_separate)

        # Combined
        bars_combined = [orange_to_orange_and_blue_to_blue_mean.mean(), orange_to_blue_mean.mean()]

        error_bars_combined = [orange_to_orange_and_blue_to_blue_error.mean(), orange_to_blue_error.mean()]

        print(orange_to_orange_and_blue_to_blue_mean.shape)
        print(orange_to_orange_mean.shape)
        print(blue_to_blue_mean.shape)
        print(orange_to_blue_mean.shape)

        # Plotting Orange, Blue, and Across Fish Groups Separately #########################################################
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar([0, 1, 2], bars_separate, width, color=["darkorange", "dodgerblue", "dimgray"])
        # Also plot the individual data points by individual relation
        
        orange_to_orange_by_individual = orange_to_orange_mean.mean(axis=0)
        blue_to_blue_by_individual = blue_to_blue_mean.mean(axis=0)
        orange_to_blue_by_individual = orange_to_blue_mean.mean(axis=0)

        print(orange_to_orange_by_individual.shape)
        print(blue_to_blue_by_individual.shape)
        print(orange_to_blue_by_individual.shape)

        ax.plot([0, 1, 2], [orange_to_orange_by_individual, blue_to_blue_by_individual, orange_to_blue_by_individual], 'k.')

        plt.errorbar([0, 1, 2], bars_separate, yerr=error_bars_separate, fmt='*', color='black',
                    markersize=0, capsize=5)

        plt.ylabel('Average Distance between Members (Units)', fontweight='bold', fontsize=14)
        plt.title(f'\n  Average Distance Between Socially-Reared Fish During Self-Segregation \n({model} = 1)  \n',
                fontweight='bold', fontsize=12)

        # xticks()
        plt.xticks([0, 1, 2], ['Orange to Orange Fish', 'Blue to Blue Fish', 'Across Fish Groups'], fontweight='bold',
                fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'{DATA_PATH}/FULL_{model}_SS.png')
        plt.close()

        # Plotting Within Group Together #################################################################################
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar([0, 1], bars_combined, width, color=["lightgray", "darkslategray"])
        
        # Also plot the individual data points by individual relation
        orange_to_orange_and_blue_to_blue_by_individual = orange_to_orange_and_blue_to_blue_mean.mean(axis=0)
        print(orange_to_orange_and_blue_to_blue_by_individual.shape)
        
        ax.plot([0], [orange_to_orange_and_blue_to_blue_by_individual], 'k.')
        ax.plot([1], [orange_to_blue_by_individual], 'k.')

        plt.errorbar([0, 1], bars_combined, yerr=error_bars_combined, fmt='*', color='black', markersize=0, capsize=5)

        plt.ylabel('Average Distance between Members (Units)', fontweight='bold', fontsize=14)
        plt.title(f'\n  Average Distance Between Socially-Reared Fish During Self-Segregation \n({model} = 1)  \n',
                fontweight='bold', fontsize=12)

        # xticks()
        plt.xticks([0, 1], ['Within Groups', 'Across Groups'], fontweight='bold',
                fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(f'{DATA_PATH}/FULL_{model}_SS_WITHIN.png')
        plt.close()
        print("[INFO] Bars Generated")

        # Save the individual data points as CSV
        individual_data = {
            'orange_to_orange': orange_to_orange_by_individual,
            'blue_to_blue': blue_to_blue_by_individual,
            'orange_to_blue': orange_to_blue_by_individual,
        }
        individual_data_df = pd.DataFrame(individual_data)
        individual_data_df.to_csv(rf"{DATA_PATH}/FULL_FISH_{model}_Self_Segregation_Individual_Data.csv", index=False)

    print("[INFO] Finished")
