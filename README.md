# Parallel Development of Social Behavior in Biological and Artificial Fish

## Joshua D McGraw, Donsuk Lee, Justin N Wood. Nature Communications 2024.

![2afc-demo](https://github.com/buildingamind/McGraw-Lee-Wood-2024/assets/76073531/08bd9f70-0a60-4438-9d9b-2f02c1b24121)

# Contents:
- All Data (Extracted from Experiment / Condition / Excel File)
- Experiment 2 - Progressive Grouping over Development
- Experiment 3 - Us vs Them in Two Colored Fishes

# **Walkthough**
This repository contins the materials and source code associated with `Parallel development of social behavior in biological and artificial fish. (McGraw, Lee, Wood 2024)`  Nature Communications.


## Raw Data is presented in 3 locations of this repository:

### 1. Figures Excel File
All Raw data associated with each figure is included under `/All Data/McGraw, Lee, Wood 2024 - Nature Communications - Final Figures Data Source.xlsx`

### 2. Raw CSV files are available in `/All Data`
The remaining files are the raw .csv file versions of the data generated from the scripts (Present also in each of the experiment directories described below)

### 3. Raw CSV files, Significance Tests, and Figures are also grouped by experiment, task, and condition

The raw data is also included according to the experiment based on Experiment 2 or 3, followed by the task, condition, and model type, (e.g. `Experiment 3 - Us vs Them/EXP3_2AFC_TASK/EXP3_2FAC_FULL_CSV/FULL_CUR_2AFC_SocialPreference_by_Episode.csv`)

<hr>

### Naming Conventions:
Among the sources files, certain naming conventions are used. You may refer to this list for definitions:
- `CUR` - Curiosity Module
- `CTR` - Contrastive Learning Module
- `CRF` - Curiosity with Random Features
- `RND` - Random Network Distillation
- `FULL` - Indicates that the strength of the Intrinsic Motivation Module is at full strength (1.0)
- `LOW` - Indicates the strength of the Intrinsic Motivation Module is at low strength (0.001)
- `ALONE` - Indicates that the agent was reared in isolation, without any social experience.
- `2AFC` - Two-Alternative Forced Choice Task
- `SS` - Self-Segregation Task

## Running ML-Agents on the provided Executables (Windows)
Executables are provided as Windows .exe files. The purpose of these executable files are to launched through Python + (https://github.com/Unity-Technologies/ml-agents)[ML-Agents].
The following versions of ML-Agents were used in the paper:
- Unity ML-Agents version 2.0.1
- Python 3.8.10 with PyTorch 1.7.1+cu110,
- Python ‘mlagents’ library version 0.26.0,
- ‘ml-agents-envs’ version 0.26.0
- ML-agents’ Communicator API 1.5.0.

Below are the steps to set up and run the executables provided, using the specified versions. These instructions assume you have a Windows OS and some familiarity with Python environments.

### Step-by-Step Guide to Set Up and Execute a .exe File Using ML-Agents

#### 1. **Set Up Your Python Environment**

1. **Install Python 3.8.10**:

   Make sure you have Python 3.8.10 installed. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3810/).

2. **Create a Virtual Environment**:

   Use `venv` to create a virtual environment to manage your dependencies.

   ```bash
   python -m venv mlagents_env
   ```

3. **Activate the Virtual Environment**:

   ```bash
   .\mlagents_env\Scripts\activate
   ```

#### 2. **Install Required Dependencies**

Once the virtual environment is activated, you need to install the specific versions of the required libraries.

1. **Upgrade pip**:

   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install ML-Agents and Other Libraries**:

   ```bash
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   pip install mlagents==0.26.0
   ```

   This will also install the `ml-agents-envs` and `ML-agents’ Communicator API` with the proper versions.

<hr>

#### 3. **Launching the Provided Executables through ML-Agents**

1. **Navigate to the Executable Directory**:

   Open your command shell and navigate to the directory where your executable (`.exe` file) is located.

   ```bash
   cd path\to\your\executable\directory
   ```

To run the provided executable (`.exe` file) using the ML-Agents toolkit, you need to use the `mlagents-learn` command. Here’s how you can do it:

#### 4. **Create/Prepare a Training Configuration**

Before running the training, you need a training configuration file (`.yaml`). This file contains the hyperparameters and settings for training the agents.

Here is a simple example of a config file (`config.yaml`). You should modify `env_path` and `run_id`:

```yaml
default_settings:
  trainer_type: ppo
  hyperparameters:
    batch_size: 256
    buffer_size: 2048
    learning_rate: 0.0003
    learning_rate_schedule: linear
  network_settings:
    normalize: false
    hidden_units: 512
    num_layers: 3
    vis_encode_type: simple
    memory: null
    goal_conditioning_type: hyper
  reward_signals:
    curiosity:
      gamma: 0.99
      strength: 1
      network_settings:
        normalize: false
        hidden_units: 128
        num_layers: 3
        vis_encode_type: simple
        memory: null
        goal_conditioning_type: hyper
  init_path: null
  keep_checkpoints: 5
  checkpoint_interval: 10000
  max_steps: 1000000
  time_horizon: 128
  summary_freq: 1000
  threaded: false
  self_play: null
  behavioral_cloning: null
env_settings:
  env_path: CONFIGURE/THIS/PATH/TO/YOUR/ENVIRONEMNT_EXECUTABLE.x86_64
  env_args: null
  base_port: 6000
  num_envs: 1
  seed: 1
engine_settings:
  width: 80
  height: 80
  quality_level: 5
  time_scale: 20.0
  target_frame_rate: -1
  capture_frame_rate: 300
  no_graphics: false
environment_parameters: null
checkpoint_settings:
  run_id: YOUR_RUN_ID
  initialize_from: null
  load_model: false
  resume: false
  force: false
  train_model: false
  inference: false
  results_dir: results
torch_settings:
  device: cuda
debug: false
```

*The model type itself can  be configured from the line stating `curiosity:`, which can be changed to `curiosity_rf:`, `rnd:`, or `contrastive:`.*

#### 5. **Run the Training Session**

Use the `mlagents-learn` command to start the training process. You will need to provide the path to your configuration file and execute the .exe file within the command:

```bash
mlagents-learn config.yaml --run-id=your_run_id --env=path\to\your\executable\your_executable.exe
```

Here is what each part of this command does:
- `config.yaml`: Specifies the configuration file for training.
- `--run-id=your_run_id`: Assigns a unique ID to this training run. You can name it anything, e.g., `experiment1`.
- `--env=path\to\your\executable\your_executable.exe`: Specifies the path to the environment executable.

#### 6. **Monitoring Training**  (Optional)

While the training is running, you can monitor it using TensorBoard (if installed through Conda). Run TensorBoard in a new Anaconda activated terminal window:

```bash
tensorboard --logdir=results
```

Open up your web browser and go to `http://localhost:6006` to visualize the training progress

#### 7. **Stopping Training**

- To stop the training immediately, you can use `Ctrl + C` in the terminal where you started the `mlagents-learn` command (not recommended). 
- Otherwise, the agents will continue to train until the `max_steps` has been reached (recommended)

### Conclusion

After running ML-Agents on the environment, a `results` folder will be exported to the path configured in your `config.yaml` file along with the `run_id` as the folder containin the trained models. Results are in `.onnx` (Open Neural Network Exchange) format, which can be directly assigned to an ML-Agent Behavior policy inside the Unity Game Engine.

If you encounter any issues, ensure that all versions match those specified, and refer to the [ML-Agents documentation](https://github.com/Unity-Technologies/ml-agents) for more detailed information. Otherwise, please open an Issue ticket from within GitHub.

## Running Example Executables of Pre-Built Curiosity Models:
- If you would like to inspect the behavior of a trained Curiosity models, you can open any `*_testing` folders and directly launch the executable without any versions of Python, PyTorch, ML-Agents, etc. installed. This is because the models have already been trained, assigned, and packaged inside the executables themselves (meaning they cannot be used for training either, because none of the ML-Agents are open for training.)
- The purpose of these environments are specifically for those who do not wish to train their own agents -- or who are not familiar with Python, Unity, etc.
- You can launch these environments simply by downloading them, extracting them, and double-clicking their .exe file (Windows)
- When you launch these environments, a folder will be created in the same executable directory under `\TwinSimulations_Data\Logs\`, which will continually output raw data concerning the position and orientation of each agent within the environment as a .csv file. (Note: In many of these environments, several tests are occuring in parallel even though the scene only shows one chamber, which is why multiple agents will show in the output CSV file.)
