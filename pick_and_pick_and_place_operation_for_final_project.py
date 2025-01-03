# importing all the required directories.
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime

class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        
        # Initializes the callback for saving the best-performing model during training.

        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        # Ensures the save directory exists.
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self):
        # Saves a checkpoint every check_freq steps.
        if self.n_calls % self.check_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_{timestamp}.zip')
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved checkpoint to {checkpoint_path}")
        return True



def make_env():
    """Create a single environment instance"""
    def _init():
        controller_config = load_composite_controller_config(controller="BASIC")
        
        env = suite.make(
            # Change from "Lift" to "PickPlace for pick and place task"
            "Lift",
            robots=["Panda"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            control_freq=180,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
        )
        
        env = GymWrapper(env)
        return Monitor(env)
    return _init




def create_env():
    # Create the robosuite environment with default Lift task.
    envs = DummyVecEnv([make_env()])
    return envs

def find_latest_model(models_dir="trained_models"):
    # Find the most recent model file in the directory
    if not os.path.exists(models_dir):
        return None
        
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        return None
        
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    return os.path.join(models_dir, latest_model)

def train_agent(total_timesteps=1_500_000, continue_training=True):
    # Train the SAC agent with automatic model loading if available.
    env = create_env()
    
    # Set up model directories.
    models_dir = "trained_models"
    logs_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Try to load the most recent model if continue_training is set to True .
    latest_model = None
    if continue_training:
        latest_model = find_latest_model(models_dir)
        if latest_model:
            print(f"\nLoading latest model: {latest_model}")
        else:
            print("\nNo existing model found. Starting fresh training.")

    if latest_model and continue_training:
        try:
            model = SAC.load(latest_model, env=env)
            print("Successfully loaded previous model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead.")
            model = create_new_model(env, logs_dir)
    else:
        model = create_new_model(env, logs_dir)

    # Create callback
    checkpoint_callback = SaveBestModelCallback(
        check_freq=5000,
        save_path=models_dir,
        verbose=1
    )
    
    try:
        print("\nStarting the training")
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=100,
            reset_num_timesteps=False
        )

        final_model_path = os.path.join(models_dir, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        model.save(final_model_path)
        print(f"\nSaved final model to {final_model_path}")
        
        return model, final_model_path
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupt_model_path = os.path.join(models_dir, f'interrupted_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        model.save(interrupt_model_path)
        print(f"Saved interrupted model to {interrupt_model_path}")
        return model, interrupt_model_path

def create_new_model(env, logs_dir):
    # Create a new SAC model with optimized hyperparameters.
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=8_000_000,
        batch_size=49152,
        tau=0.005,
        gamma=0.95,
        ent_coef="auto_0.1",
        train_freq=3,
        gradient_steps=6,
        learning_starts=5000,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 512],
                qf=[1024, 1024, 512]
            ),
            use_expln=True,
        ),
        tensorboard_log=logs_dir,
        verbose=1,
        device="cuda",
    )

def evaluate_and_demo(model_path=None, n_eval_episodes=5):
    # Evaluate and demonstrate the trained model.
    # Create a single environment for evaluation
    env = make_env()()
    
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No trained model found. Please train the agent first.")
            return
    
    print(f"\nLoading model from {model_path}")
    model = SAC.load(model_path)
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    print(f"\nMean reward over {n_eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    print("\nRunning demonstration episodes...")
    for episode in range(3):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
            step += 1
            
        print(f"Demo episode {episode + 1} finished with reward: {episode_reward:.2f} in {step} steps")

if __name__ == "__main__":
    # Custom interface for easier option selection.
    print("Choose an option:")
    print("1. Continue training from latest model (if available)")
    print("2. Start fresh training")
    print("3. Evaluate latest model")
    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        model, model_path = train_agent(total_timesteps=1_500_000, continue_training=True)
    elif choice == "2":
        model, model_path = train_agent(total_timesteps=1_500_000, continue_training=False)
    elif choice == "3":
        evaluate_and_demo()
    else:
        # this is to not crash the code.
        print("Invalid choice. Please run again with a valid option.")