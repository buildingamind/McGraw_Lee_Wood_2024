import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy import stats

for model in ['CUR', 'RND', 'CRF', 'CTR']:
    # Load the CSV file into a Pandas DataFrame
    file_path = rf"G:/McGraw 2024 Source/Experiment 2 - Progressive Grouping over Development/EXP2_Underwater_Test/EXP2 Training Logs/ALL_{model}.csv"
    df = pd.read_csv(file_path, on_bad_lines='skip')

    print(df.head())

    # Initialize lists to store the average and standard deviation of pairwise distances per episode
    orange_average_pairwise_distances = []
    blue_average_pairwise_distances = []
    orange_std_pairwise_distances = []
    blue_std_pairwise_distances = []

    # Get the unique list of episodes
    episodes = df['episode'].unique()[:100]

    # Loop through each episode and calculate pairwise distances
    for episode in episodes:
        # Filter the DataFrame for the current episode
        episode_df = df[df['episode'] == episode]
        
        # Extract positions for all 16 clownfish agents
        orange_positions = []
        blue_positions = []
        for i in range(1, 17):
            orange_agent_prefix = f'orangeclownfish_{i:02d}'
            blue_agent_prefix = f'blueclownfish_{i:02d}'
            orange_agent_positions = episode_df[[f'{orange_agent_prefix}.xposition', f'{orange_agent_prefix}.yposition', f'{orange_agent_prefix}.zposition']].values
            blue_agent_positions = episode_df[[f'{blue_agent_prefix}.xposition', f'{blue_agent_prefix}.yposition', f'{blue_agent_prefix}.zposition']].values
            orange_positions.append(orange_agent_positions)
            blue_positions.append(blue_agent_positions)
            
        # Stack all positions into a single numpy array
        orange_agent_positions_stack = np.stack(orange_positions, axis=1).reshape(-1, 16, 3)
        blue_agent_positions_stack = np.stack(blue_positions, axis=1).reshape(-1, 16, 3)
        
        # Calculate pairwise distances for each step
        orange_pairwise_distances = [pdist(step_pos) for step_pos in orange_agent_positions_stack]
        blue_pairwise_distances = [pdist(step_pos) for step_pos in blue_agent_positions_stack]
        
        # Calculate the average and standard deviation of pairwise distances per step in the episode:
        # Orange
        orange_average_episode_distances = [np.mean(distances) for distances in orange_pairwise_distances]
        orange_std_episode_distances = [stats.sem(distances) for distances in orange_pairwise_distances]
        # Blue
        blue_average_episode_distances = [np.mean(distances) for distances in blue_pairwise_distances]
        blue_std_episode_distances = [stats.sem(distances) for distances in blue_pairwise_distances]
        
        # Calculate the mean and standard deviation of these average distances for the entire episode
        orange_mean_of_averages = np.mean(orange_average_episode_distances)
        orange_std_of_averages = np.mean(orange_std_episode_distances)
        blue_mean_of_averages = np.mean(blue_average_episode_distances)
        blue_std_of_averages = np.mean(blue_std_episode_distances)
        
        # Append to the lists
        orange_average_pairwise_distances.append(orange_mean_of_averages)
        orange_std_pairwise_distances.append(orange_std_of_averages)
        blue_average_pairwise_distances.append(blue_mean_of_averages)
        blue_std_pairwise_distances.append(blue_std_of_averages)

    # Plotting the average pairwise distance over episodes with standard error as error bars
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    ax.set_title(f'Average Pairwise Distance Between Group During Training ({model})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Pairwise Distance')
    ax.grid(True)

    episodes_range = np.arange(len(episodes))
    # ax.errorbar(episodes_range, orange_average_pairwise_distances, yerr=orange_std_pairwise_distances, color="orange", ecolor='orange', capsize=0, label='Orange Clownfish')
    ax.plot(episodes_range, orange_average_pairwise_distances, color="orange")
    ax.plot(episodes_range, blue_average_pairwise_distances, color="cornflowerblue")

    # ax.errorbar(episodes_range, blue_average_pairwise_distances, yerr=blue_std_pairwise_distances, color="cornflowerblue", ecolor='cornflowerblue', capsize=0, label='Blue Clownfish')
    ax.fill_between(episodes_range, np.subtract(orange_average_pairwise_distances, orange_std_pairwise_distances), np.add(orange_average_pairwise_distances, orange_std_pairwise_distances), color="orange", alpha=0.2)
    ax.fill_between(episodes_range, np.subtract(blue_average_pairwise_distances, blue_std_pairwise_distances), np.add(blue_average_pairwise_distances, blue_std_pairwise_distances), color="cornflowerblue", alpha=0.2)

    ax.legend(['Orange Clownfish', 'Blue Clownfish'], loc='upper right')

    plt.savefig(f"G:/McGraw 2024 Source/Experiment 2 - Progressive Grouping over Development/EXP2_Underwater_Test/EXP2 Training Logs/{model}_Average_Pairwise_Distance.png")
    

    # Export full data as a CSV file
    data = {
        'episode': episodes,
        'orange_average_pairwise_distances': orange_average_pairwise_distances,
        'orange_std_pairwise_distances': orange_std_pairwise_distances,
        'blue_average_pairwise_distances': blue_average_pairwise_distances,
        'blue_std_pairwise_distances': blue_std_pairwise_distances
    }
    full_data_df = pd.DataFrame(data)
    full_data_df.to_csv(f"G:/McGraw 2024 Source/Experiment 2 - Progressive Grouping over Development/EXP2_Underwater_Test/EXP2 Training Logs/{model}_Average_Pairwise_Distances.csv", index=False)