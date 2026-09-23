from fog_env import Offload
from RL_brain import DeepQNetwork

import numpy as np
import random
import pickle
import os
import time
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"

SAVE_EVERY = 5

np.set_printoptions(
    threshold=np.inf
)


# ============================================================
# GPU / CPU INFORMATION
# ============================================================

def print_device_information():

    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    print(
        "TensorFlow version:",
        tf.__version__
    )

    cpus = tf.config.list_physical_devices(
        "CPU"
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        "CPU:",
        cpus
    )

    print(
        "GPU:",
        gpus
    )

    if len(gpus) > 0:

        print(
            "GPU detected: YES"
        )

        for gpu in gpus:

            print(
                "GPU device:",
                gpu
            )

    else:

        print(
            "GPU detected: NO"
        )

        print(
            "Training will use CPU."
        )

    print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    iot_RL_list,
    episode,
    final=False
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    data = {

        "episode": episode,

        "reward": [
            rl.reward_store
            for rl in iot_RL_list
        ],

        "action": [
            rl.action_store
            for rl in iot_RL_list
        ],

        "delay": [
            rl.delay_store
            for rl in iot_RL_list
        ]
    }

    if final:

        filename = (
            "results_final.pkl"
        )

    else:

        filename = (
            f"results_ep{episode + 1}.pkl"
        )

    filepath = os.path.join(
        RESULTS_DIR,
        filename
    )

    with open(
        filepath,
        "wb"
    ) as f:

        pickle.dump(
            data,
            f
        )

    print(
        f"Results saved: {filepath}"
    )


# ============================================================
# REWARD FUNCTION
# ============================================================

def reward_fun(
    delay,
    max_delay,
    unfinish_indi
):

    penalty = -max_delay * 2

    if unfinish_indi:

        reward = penalty

    else:

        reward = -delay

    return reward


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train(
    env,
    iot_RL_list,
    NUM_EPISODE
):

    RL_step = 0

    total_training_start = time.perf_counter()

    # ========================================================
    # EPISODE LOOP
    # ========================================================

    for episode in range(
        NUM_EPISODE
    ):

        episode_start = time.perf_counter()

        print()
        print("=" * 60)

        print(
            f"Episode {episode + 1}/{NUM_EPISODE}"
        )

        print(
            "Epsilon:",
            iot_RL_list[0].epsilon
        )

        # ====================================================
        # GENERATE BITRATE ARRIVAL
        # ====================================================

        bitarrive = np.random.uniform(
            env.min_bit_arrive,
            env.max_bit_arrive,
            size=(
                env.n_time,
                env.n_iot
            )
        )

        task_prob = (
            env.task_arrive_prob
        )

        bitarrive = (
            bitarrive
            *
            (
                np.random.uniform(
                    0,
                    1,
                    size=(
                        env.n_time,
                        env.n_iot
                    )
                )
                < task_prob
            )
        )

        # Last MAX_DELAY slots are empty
        bitarrive[
            -env.max_delay:,
            :
        ] = np.zeros(
            (
                env.max_delay,
                env.n_iot
            )
        )

        # ====================================================
        # OBSERVATION HISTORY
        # ====================================================

        history = []

        for time_index in range(
            env.n_time
        ):

            history.append([])

            for iot_index in range(
                env.n_iot
            ):

                tmp_dict = {

                    "observation":
                        np.zeros(
                            env.n_features
                        ),

                    "lstm":
                        np.zeros(
                            env.n_lstm_state
                        ),

                    "action":
                        np.nan,

                    "observation_":
                        np.zeros(
                            env.n_features
                        ),

                    "lstm_":
                        np.zeros(
                            env.n_lstm_state
                        )
                }

                history[
                    time_index
                ].append(
                    tmp_dict
                )

        # ====================================================
        # REWARD INDICATOR
        # ====================================================

        reward_indicator = np.zeros(
            (
                env.n_time,
                env.n_iot
            )
        )

        # ====================================================
        # INITIALIZE ENVIRONMENT
        # ====================================================

        (
            observation_all,
            lstm_state_all
        ) = env.reset(
            bitarrive
        )

        # ====================================================
        # TIME SLOT LOOP
        # ====================================================

        while True:

            # =================================================
            # ACTION
            # =================================================

            action_all = np.zeros(
                env.n_iot
            )

            for iot_index in range(
                env.n_iot
            ):

                observation = np.squeeze(
                    observation_all[
                        iot_index,
                        :
                    ]
                )

                # ---------------------------------------------
                # No task
                # ---------------------------------------------

                if np.sum(
                    observation
                ) == 0:

                    action_all[
                        iot_index
                    ] = 0

                # ---------------------------------------------
                # Task exists
                # ---------------------------------------------

                else:

                    action_all[
                        iot_index
                    ] = (
                        iot_RL_list[
                            iot_index
                        ].choose_action(
                            observation
                        )
                    )

                # ---------------------------------------------
                # Store action
                # ---------------------------------------------

                if observation[0] != 0:

                    iot_RL_list[
                        iot_index
                    ].do_store_action(
                        episode,
                        env.time_count,
                        action_all[
                            iot_index
                        ]
                    )

            # =================================================
            # ENVIRONMENT STEP
            # =================================================

            (
                observation_all_,
                lstm_state_all_,
                done
            ) = env.step(
                action_all
            )

            # =================================================
            # UPDATE LSTM
            # =================================================

            for iot_index in range(
                env.n_iot
            ):

                iot_RL_list[
                    iot_index
                ].update_lstm(
                    lstm_state_all_[
                        iot_index,
                        :
                    ]
                )

            # =================================================
            # GET DELAY
            # =================================================

            process_delay = (
                env.process_delay
            )

            unfinish_indi = (
                env.process_delay_unfinish_ind
            )

            # =================================================
            # STORE TRANSITIONS
            # =================================================

            for iot_index in range(
                env.n_iot
            ):

                history[
                    env.time_count - 1
                ][iot_index][
                    "observation"
                ] = observation_all[
                    iot_index,
                    :
                ]

                history[
                    env.time_count - 1
                ][iot_index][
                    "lstm"
                ] = np.squeeze(
                    lstm_state_all[
                        iot_index,
                        :
                    ]
                )

                history[
                    env.time_count - 1
                ][iot_index][
                    "action"
                ] = action_all[
                    iot_index
                ]

                history[
                    env.time_count - 1
                ][iot_index][
                    "observation_"
                ] = observation_all_[
                    iot_index
                ]

                history[
                    env.time_count - 1
                ][iot_index][
                    "lstm_"
                ] = np.squeeze(
                    lstm_state_all_[
                        iot_index,
                        :
                    ]
                )

                # ---------------------------------------------
                # Find completed tasks
                # ---------------------------------------------

                update_index = np.where(
                    (
                        1
                        - reward_indicator[
                            :,
                            iot_index
                        ]
                    )
                    *
                    process_delay[
                        :,
                        iot_index
                    ]
                    > 0
                )[0]

                # ---------------------------------------------
                # Store each completed transition
                # ---------------------------------------------

                if len(update_index) != 0:

                    for update_ii in range(
                        len(update_index)
                    ):

                        time_index = (
                            update_index[
                                update_ii
                            ]
                        )

                        reward = reward_fun(
                            process_delay[
                                time_index,
                                iot_index
                            ],

                            env.max_delay,

                            unfinish_indi[
                                time_index,
                                iot_index
                            ]
                        )

                        # -------------------------------------
                        # Experience replay
                        # -------------------------------------

                        iot_RL_list[
                            iot_index
                        ].store_transition(

                            history[
                                time_index
                            ][iot_index][
                                "observation"
                            ],

                            history[
                                time_index
                            ][iot_index][
                                "lstm"
                            ],

                            history[
                                time_index
                            ][iot_index][
                                "action"
                            ],

                            reward,

                            history[
                                time_index
                            ][iot_index][
                                "observation_"
                            ],

                            history[
                                time_index
                            ][iot_index][
                                "lstm_"
                            ]
                        )

                        # -------------------------------------
                        # Store reward
                        # -------------------------------------

                        iot_RL_list[
                            iot_index
                        ].do_store_reward(

                            episode,

                            time_index,

                            reward
                        )

                        # -------------------------------------
                        # Store delay
                        # -------------------------------------

                        iot_RL_list[
                            iot_index
                        ].do_store_delay(

                            episode,

                            time_index,

                            process_delay[
                                time_index,
                                iot_index
                            ]
                        )

                        reward_indicator[
                            time_index,
                            iot_index
                        ] = 1

            # =================================================
            # INCREASE RL STEP
            # =================================================

            RL_step += 1

            # =================================================
            # UPDATE OBSERVATION
            # =================================================

            observation_all = (
                observation_all_
            )

            lstm_state_all = (
                lstm_state_all_
            )

            # =================================================
            # LEARNING
            # =================================================

            if (
                RL_step > 200
                and RL_step % 10 == 0
            ):

                for iot in range(
                    env.n_iot
                ):

                    iot_RL_list[
                        iot
                    ].learn()

            # =================================================
            # END EPISODE
            # =================================================

            if done:

                break

        # ====================================================
        # EPISODE TIME
        # ====================================================

        episode_time = (
            time.perf_counter()
            - episode_start
        )

        print(
            f"Episode time: "
            f"{episode_time:.2f} seconds"
        )

        print(
            f"Total RL steps: "
            f"{RL_step}"
        )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if (
            (episode + 1)
            % SAVE_EVERY
            == 0
        ):

            save_results(
                iot_RL_list,
                episode
            )

            # -----------------------------------------------
            # Save neural network weights
            # -----------------------------------------------

            model_dir = os.path.join(
                RESULTS_DIR,
                "models",
                f"episode_{episode + 1}"
            )

            os.makedirs(
                model_dir,
                exist_ok=True
            )

            for iot in range(
                env.n_iot
            ):

                filepath = os.path.join(
                    model_dir,
                    f"iot_{iot}.weights.h5"
                )

                iot_RL_list[
                    iot
                ].save_model(
                    filepath
                )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    total_training_time = (
        time.perf_counter()
        - total_training_start
    )

    print()
    print("=" * 60)

    print(
        "TRAINING FINISHED"
    )

    print(
        f"Total training time: "
        f"{total_training_time:.2f} seconds"
    )

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Total training time: "
        f"{total_training_time / 3600:.2f} hours"
    )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    save_results(
        iot_RL_list,
        NUM_EPISODE - 1,
        final=True
    )

    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    final_model_dir = os.path.join(
        RESULTS_DIR,
        "models",
        "final"
    )

    os.makedirs(
        final_model_dir,
        exist_ok=True
    )

    for iot in range(
        env.n_iot
    ):

        filepath = os.path.join(
            final_model_dir,
            f"iot_{iot}.weights.h5"
        )

        iot_RL_list[
            iot
        ].save_model(
            filepath
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # PRINT HARDWARE
    # ========================================================

    print_device_information()

    # ========================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================

    NUM_IOT = 50

    NUM_FOG = 5

    NUM_EPISODE = 10

    NUM_TIME_BASE = 100

    MAX_DELAY = 10

    NUM_TIME = (
        NUM_TIME_BASE
        + MAX_DELAY
    )

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    env = Offload(
        NUM_IOT,
        NUM_FOG,
        NUM_TIME,
        MAX_DELAY
    )

    # ========================================================
    # CREATE DQN AGENTS
    # ========================================================

    iot_RL_list = []

    for iot in range(
        NUM_IOT
    ):

        print(
            f"Creating DQN agent "
            f"{iot + 1}/{NUM_IOT}"
        )

        agent = DeepQNetwork(

            env.n_actions,

            env.n_features,

            env.n_lstm_state,

            env.n_time,

            learning_rate=0.01,

            reward_decay=0.9,

            e_greedy=0.99,

            replace_target_iter=200,

            memory_size=500,

            batch_size=32,

            e_greedy_increment=0.00025,

            n_lstm_step=10,

            dueling=True,

            double_q=True,

            N_L1=20,

            N_lstm=20
        )

        iot_RL_list.append(
            agent
        )

    # ========================================================
    # START TRAINING
    # ========================================================

    train(
        env,
        iot_RL_list,
        NUM_EPISODE
    )

    print(
        "Training Finished"
    )