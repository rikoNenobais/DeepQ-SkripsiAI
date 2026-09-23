import numpy as np
import tensorflow as tf
from collections import deque


class DeepQNetwork:

    def __init__(
        self,
        n_actions,
        n_features,
        n_lstm_features,
        n_time,
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
    ):

        # ============================================================
        # BASIC CONFIGURATION
        # ============================================================

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time

        self.lr = learning_rate
        self.gamma = reward_decay

        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment

        self.epsilon = (
            0.0
            if e_greedy_increment is not None
            else self.epsilon_max
        )

        self.replace_target_iter = replace_target_iter

        self.memory_size = memory_size
        self.batch_size = batch_size

        self.dueling = dueling
        self.double_q = double_q

        self.learn_step_counter = 0

        # ============================================================
        # NETWORK CONFIGURATION
        # ============================================================

        self.N_L1 = N_L1
        self.N_lstm = N_lstm

        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features

        # ============================================================
        # EXPERIENCE REPLAY MEMORY
        #
        # Original structure:
        #
        # [s, action, reward, s_,
        #  lstm_s, lstm_s_]
        # ============================================================

        self.memory = np.zeros(
            (
                self.memory_size,
                self.n_features
                + 1
                + 1
                + self.n_features
                + self.n_lstm_state
                + self.n_lstm_state
            ),
            dtype=np.float32
        )

        self.memory_counter = 0

        # ============================================================
        # BUILD EVALUATION AND TARGET NETWORK
        # ============================================================

        self.eval_net = self._build_network(name="eval_net")
        self.target_net = self._build_network(name="target_net")

        # ============================================================
        # OPTIMIZER
        # ============================================================

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.lr
        )

        # ============================================================
        # RESULT STORAGE
        # ============================================================

        self.reward_store = []
        self.action_store = []
        self.delay_store = []

        # ============================================================
        # LSTM HISTORY
        # ============================================================

        self.lstm_history = deque(
            maxlen=self.n_lstm_step
        )

        for _ in range(self.n_lstm_step):
            self.lstm_history.append(
                np.zeros(
                    self.n_lstm_state,
                    dtype=np.float32
                )
            )

        # ============================================================
        # Q VALUE STORAGE
        # ============================================================

        self.store_q_value = []

        # ============================================================
        # INITIAL TARGET NETWORK
        # ============================================================

        self.update_target_network()

    # =================================================================
    # BUILD NETWORK
    # =================================================================

    def _build_network(self, name):

        # -------------------------------------------------------------
        # INPUT 1: CURRENT OBSERVATION
        # Shape:
        # (batch, n_features)
        # -------------------------------------------------------------

        observation_input = tf.keras.Input(
            shape=(self.n_features,),
            name=f"{name}_observation"
        )

        # -------------------------------------------------------------
        # INPUT 2: LSTM HISTORY
        #
        # Shape:
        # (batch, n_lstm_step, n_lstm_state)
        # -------------------------------------------------------------

        lstm_input = tf.keras.Input(
            shape=(
                self.n_lstm_step,
                self.n_lstm_state
            ),
            name=f"{name}_lstm"
        )

        # -------------------------------------------------------------
        # LSTM
        # -------------------------------------------------------------

        lstm_output = tf.keras.layers.LSTM(
            self.N_lstm,
            return_sequences=False,
            name=f"{name}_lstm_layer"
        )(lstm_input)

        # -------------------------------------------------------------
        # CONCATENATE LSTM OUTPUT + CURRENT STATE
        # -------------------------------------------------------------

        x = tf.keras.layers.Concatenate(
            name=f"{name}_concat"
        )([
            lstm_output,
            observation_input
        ])

        # -------------------------------------------------------------
        # FIRST DENSE LAYER
        # -------------------------------------------------------------

        x = tf.keras.layers.Dense(
            self.N_L1,
            activation="relu",
            name=f"{name}_dense1"
        )(x)

        # -------------------------------------------------------------
        # SECOND DENSE LAYER
        # -------------------------------------------------------------

        x = tf.keras.layers.Dense(
            self.N_L1,
            activation="relu",
            name=f"{name}_dense2"
        )(x)

        # =============================================================
        # DUELING DQN
        # =============================================================

        if self.dueling:

            # ---------------------------------------------------------
            # VALUE STREAM
            # V(s)
            # ---------------------------------------------------------

            value = tf.keras.layers.Dense(
                1,
                activation=None,
                name=f"{name}_value"
            )(x)

            # ---------------------------------------------------------
            # ADVANTAGE STREAM
            # A(s,a)
            # ---------------------------------------------------------

            advantage = tf.keras.layers.Dense(
                self.n_actions,
                activation=None,
                name=f"{name}_advantage"
            )(x)

            # ---------------------------------------------------------
            # Q(s,a) = V(s) + A(s,a) - mean(A)
            # ---------------------------------------------------------

            advantage_mean = tf.keras.layers.Lambda(
                lambda a: tf.reduce_mean(
                    a,
                    axis=1,
                    keepdims=True
                ),
                name=f"{name}_advantage_mean"
            )(advantage)

            q_values = tf.keras.layers.Add(
                name=f"{name}_q_values"
            )([
                value,
                tf.keras.layers.Subtract()([
                    advantage,
                    advantage_mean
                ])
            ])

        # =============================================================
        # STANDARD DQN
        # =============================================================

        else:

            q_values = tf.keras.layers.Dense(
                self.n_actions,
                activation=None,
                name=f"{name}_q_values"
            )(x)

        # -------------------------------------------------------------
        # CREATE MODEL
        # -------------------------------------------------------------

        model = tf.keras.Model(
            inputs=[
                observation_input,
                lstm_input
            ],
            outputs=q_values,
            name=name
        )

        return model

    # =================================================================
    # UPDATE TARGET NETWORK
    # =================================================================

    def update_target_network(self):

        self.target_net.set_weights(
            self.eval_net.get_weights()
        )

    # =================================================================
    # STORE TRANSITION
    # =================================================================

    def store_transition(
        self,
        s,
        lstm_s,
        a,
        r,
        s_,
        lstm_s_
    ):

        transition = np.hstack(
            (
                s,
                [a],
                [r],
                s_,
                lstm_s,
                lstm_s_
            )
        )

        index = (
            self.memory_counter
            % self.memory_size
        )

        self.memory[index, :] = transition

        self.memory_counter += 1

    # =================================================================
    # UPDATE LSTM HISTORY
    # =================================================================

    def update_lstm(self, lstm_s):

        self.lstm_history.append(
            np.asarray(
                lstm_s,
                dtype=np.float32
            )
        )

    # =================================================================
    # CHOOSE ACTION
    # =================================================================

    def choose_action(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        observation = observation[np.newaxis, :]

        # -------------------------------------------------------------
        # EPSILON-GREEDY
        # -------------------------------------------------------------

        if np.random.uniform() < self.epsilon:

            # ---------------------------------------------------------
            # GET LSTM HISTORY
            # ---------------------------------------------------------

            lstm_observation = np.asarray(
                self.lstm_history,
                dtype=np.float32
            )

            lstm_observation = lstm_observation.reshape(
                1,
                self.n_lstm_step,
                self.n_lstm_state
            )

            # ---------------------------------------------------------
            # FORWARD PASS
            # ---------------------------------------------------------

            actions_value = self.eval_net(
                [
                    observation,
                    lstm_observation
                ],
                training=False
            ).numpy()

            # ---------------------------------------------------------
            # STORE Q VALUE
            # ---------------------------------------------------------

            self.store_q_value.append(
                {
                    "observation": observation.copy(),
                    "q_value": actions_value.copy()
                }
            )

            # ---------------------------------------------------------
            # GREEDY ACTION
            # ---------------------------------------------------------

            action = int(
                np.argmax(actions_value)
            )

        else:

            # ---------------------------------------------------------
            # RANDOM ACTION
            # ---------------------------------------------------------

            action = np.random.randint(
                0,
                self.n_actions
            )

        return action

    # =================================================================
    # LEARN
    # =================================================================

    def learn(self):

        # -------------------------------------------------------------
        # CHECK MEMORY
        # -------------------------------------------------------------

        if self.memory_counter < self.batch_size + self.n_lstm_step:
            return

        # -------------------------------------------------------------
        # UPDATE TARGET NETWORK
        # -------------------------------------------------------------

        if (
            self.learn_step_counter
            % self.replace_target_iter
            == 0
        ):

            self.update_target_network()

            print(
                "\nTarget network updated\n"
            )

        # -------------------------------------------------------------
        # DETERMINE AVAILABLE MEMORY
        # -------------------------------------------------------------

        max_memory_index = min(
            self.memory_counter,
            self.memory_size
        )

        valid_max_index = (
            max_memory_index
            - self.n_lstm_step
        )

        if valid_max_index <= 0:
            return

        # -------------------------------------------------------------
        # SAMPLE BATCH
        # -------------------------------------------------------------

        replace = (
            valid_max_index
            < self.batch_size
        )

        sample_index = np.random.choice(
            valid_max_index,
            size=self.batch_size,
            replace=replace
        )

        # -------------------------------------------------------------
        # CURRENT STATE
        #
        # [s, action, reward, s_]
        # -------------------------------------------------------------

        state_end = (
            self.n_features
            + 1
            + 1
            + self.n_features
        )

        batch_memory = self.memory[
            sample_index,
            :state_end
        ]

        # -------------------------------------------------------------
        # LSTM MEMORY
        # -------------------------------------------------------------

        lstm_batch_memory = np.zeros(
            (
                self.batch_size,
                self.n_lstm_step,
                self.n_lstm_state * 2
            ),
            dtype=np.float32
        )

        for ii in range(
            len(sample_index)
        ):

            for jj in range(
                self.n_lstm_step
            ):

                index = (
                    sample_index[ii]
                    + jj
                )

                # Safety check
                if index >= max_memory_index:
                    index = (
                        max_memory_index - 1
                    )

                lstm_batch_memory[
                    ii,
                    jj,
                    :
                ] = self.memory[
                    index,
                    state_end:
                ]

        # -------------------------------------------------------------
        # SPLIT CURRENT AND NEXT LSTM STATE
        # -------------------------------------------------------------

        lstm_current = (
            lstm_batch_memory[
                :,
                :,
                :self.n_lstm_state
            ]
        )

        lstm_next = (
            lstm_batch_memory[
                :,
                :,
                self.n_lstm_state:
            ]
        )

        # -------------------------------------------------------------
        # CURRENT STATE
        # -------------------------------------------------------------

        states = batch_memory[
            :,
            :self.n_features
        ].astype(np.float32)

        # -------------------------------------------------------------
        # NEXT STATE
        # -------------------------------------------------------------

        next_states = batch_memory[
            :,
            -self.n_features:
        ].astype(np.float32)

        # -------------------------------------------------------------
        # ACTION
        # -------------------------------------------------------------

        actions = batch_memory[
            :,
            self.n_features
        ].astype(np.int32)

        # -------------------------------------------------------------
        # REWARD
        # -------------------------------------------------------------

        rewards = batch_memory[
            :,
            self.n_features + 1
        ].astype(np.float32)

        # =============================================================
        # DOUBLE DQN
        # =============================================================

        # -------------------------------------------------------------
        # Q values from evaluation network
        # for NEXT STATE
        # -------------------------------------------------------------

        q_eval_next = self.eval_net(
            [
                next_states,
                lstm_next
            ],
            training=False
        ).numpy()

        # -------------------------------------------------------------
        # Q values from target network
        # for NEXT STATE
        # -------------------------------------------------------------

        q_target_next = self.target_net(
            [
                next_states,
                lstm_next
            ],
            training=False
        ).numpy()

        # -------------------------------------------------------------
        # TARGET Q
        # -------------------------------------------------------------

        q_target = self.eval_net(
            [
                states,
                lstm_current
            ],
            training=False
        ).numpy()

        # -------------------------------------------------------------
        # DOUBLE Q-LEARNING
        # -------------------------------------------------------------

        if self.double_q:

            # Select action using EVAL network
            best_next_actions = np.argmax(
                q_eval_next,
                axis=1
            )

            # Evaluate selected action using TARGET network
            selected_q_next = q_target_next[
                np.arange(self.batch_size),
                best_next_actions
            ]

        else:

            selected_q_next = np.max(
                q_target_next,
                axis=1
            )

        # -------------------------------------------------------------
        # BELLMAN EQUATION
        #
        # Q_target = reward + gamma * Q_next
        # -------------------------------------------------------------

        target_values = (
            rewards
            + self.gamma * selected_q_next
        )

        # -------------------------------------------------------------
        # UPDATE ONLY THE ACTION THAT WAS TAKEN
        # -------------------------------------------------------------

        q_target[
            np.arange(self.batch_size),
            actions
        ] = target_values

        # =============================================================
        # GRADIENT UPDATE
        # =============================================================

        states_tensor = tf.convert_to_tensor(
            states,
            dtype=tf.float32
        )

        lstm_current_tensor = tf.convert_to_tensor(
            lstm_current,
            dtype=tf.float32
        )

        q_target_tensor = tf.convert_to_tensor(
            q_target,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            q_eval = self.eval_net(
                [
                    states_tensor,
                    lstm_current_tensor
                ],
                training=True
            )

            loss = tf.reduce_mean(
                tf.square(
                    q_target_tensor
                    - q_eval
                )
            )

        gradients = tape.gradient(
            loss,
            self.eval_net.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.eval_net.trainable_variables
            )
        )

        # -------------------------------------------------------------
        # STORE LOSS
        # -------------------------------------------------------------

        self.cost = float(
            loss.numpy()
        )

        # -------------------------------------------------------------
        # INCREASE EPSILON
        # -------------------------------------------------------------

        if self.epsilon < self.epsilon_max:

            self.epsilon += (
                self.epsilon_increment
            )

            self.epsilon = min(
                self.epsilon,
                self.epsilon_max
            )

        # -------------------------------------------------------------
        # LEARNING STEP
        # -------------------------------------------------------------

        self.learn_step_counter += 1

    # =================================================================
    # STORE REWARD
    # =================================================================

    def do_store_reward(
        self,
        episode,
        time,
        reward
    ):

        while episode >= len(
            self.reward_store
        ):

            self.reward_store.append(
                np.zeros(
                    self.n_time
                )
            )

        self.reward_store[
            episode
        ][time] = reward

    # =================================================================
    # STORE ACTION
    # =================================================================

    def do_store_action(
        self,
        episode,
        time,
        action
    ):

        while episode >= len(
            self.action_store
        ):

            self.action_store.append(
                -np.ones(
                    self.n_time
                )
            )

        self.action_store[
            episode
        ][time] = action

    # =================================================================
    # STORE DELAY
    # =================================================================

    def do_store_delay(
        self,
        episode,
        time,
        delay
    ):

        while episode >= len(
            self.delay_store
        ):

            self.delay_store.append(
                np.zeros(
                    self.n_time
                )
            )

        self.delay_store[
            episode
        ][time] = delay

    # =================================================================
    # SAVE MODEL
    # =================================================================

    def save_model(
        self,
        filepath
    ):

        self.eval_net.save_weights(
            filepath
        )

    # =================================================================
    # LOAD MODEL
    # =================================================================

    def load_model(
        self,
        filepath
    ):

        self.eval_net.load_weights(
            filepath
        )

        self.update_target_network()