// source: https://raw.githubusercontent.com/coreylowman/dfdx/main/examples/rl-dqn.rs
use dfdx::{
    nn,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};

// use std::{
//     fs::{self, File, OpenOptions},
//     io::{self, BufRead, Write},
// };

use crate::{
    mdp::{Agent, State},
    strategy::{explore::ExplorationStrategy, terminate::TerminationStrategy},
};

const BATCH: usize = 512;
const TARGET_UPDATE_FREQ: usize = 100; // ターゲットネットワークの更新頻度

type QNetwork<const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize> = (
    (Linear<STATE_SIZE, INNER_SIZE>, ReLU),
    (Linear<INNER_SIZE, INNER_SIZE>, ReLU),
    (Linear<INNER_SIZE,INNER_SIZE>,ReLU),
    Linear<INNER_SIZE, ACTION_SIZE>,
);

type QNetworkDevice<const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize> = (
    (nn::modules::Linear<STATE_SIZE, INNER_SIZE, f32, Cpu>, ReLU),
    (nn::modules::Linear<INNER_SIZE, INNER_SIZE, f32, Cpu>, ReLU),
    (nn::modules::Linear<INNER_SIZE,INNER_SIZE,f32,Cpu>,ReLU),
    nn::modules::Linear<INNER_SIZE, ACTION_SIZE, f32, Cpu>,
);

/// An `DQNAgentTrainer` can be trained for using a certain [Agent](mdp/trait.Agent.html). After
/// training, the `DQNAgentTrainer` contains learned knowledge about the process, and can be queried
/// for this. For example, you can ask the `DQNAgentTrainer` the expected values of all possible
/// actions in a given state.
///
/// The code is partially taken from https://github.com/coreylowman/dfdx/blob/main/examples/rl-dqn.rs.

use rand::seq::SliceRandom;
use rand::thread_rng;

struct ReplayBuffer<const STATE_SIZE: usize, const ACTION_SIZE: usize, const BUFFER_SIZE: usize> {
    states: Vec<[f32; STATE_SIZE]>,
    actions: Vec<[f32; ACTION_SIZE]>,
    rewards: Vec<f32>,
    next_states: Vec<[f32; STATE_SIZE]>,
    dones: Vec<bool>,
    index: usize,
    full: bool,
}

impl<const STATE_SIZE: usize, const ACTION_SIZE: usize, const BUFFER_SIZE: usize> ReplayBuffer<STATE_SIZE, ACTION_SIZE, BUFFER_SIZE> {
    pub fn new() -> Self {
        Self {
            states: Vec::with_capacity(BUFFER_SIZE),
            actions: Vec::with_capacity(BUFFER_SIZE),
            rewards: Vec::with_capacity(BUFFER_SIZE),
            next_states: Vec::with_capacity(BUFFER_SIZE),
            dones: Vec::with_capacity(BUFFER_SIZE),
            index: 0,
            full: false,
        }
    }

    pub fn add(&mut self, state: [f32; STATE_SIZE], action: [f32; ACTION_SIZE], reward: f32, next_state: [f32; STATE_SIZE], done: bool) {
        if self.states.len() < BUFFER_SIZE {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
        } else {
            self.states[self.index] = state;
            self.actions[self.index] = action;
            self.rewards[self.index] = reward;
            self.next_states[self.index] = next_state;
            self.dones[self.index] = done;
        }

        self.index = (self.index + 1) % BUFFER_SIZE;
        if self.index == 0 {
            self.full = true;
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<([f32; STATE_SIZE], [f32; ACTION_SIZE], f32, [f32; STATE_SIZE], bool)> {
        let mut rng = thread_rng();
        let sample_size = if self.full { BUFFER_SIZE } else { self.index };
        let mut indices: Vec<usize> = (0..sample_size).collect();
        indices.shuffle(&mut rng);

        indices.into_iter().take(batch_size).map(|i| {
            (self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i])
        }).collect()
    }
}

fn unzip_n<const STATE_SIZE: usize, const ACTION_SIZE: usize>(
    vec: Vec<([f32; STATE_SIZE], [f32; ACTION_SIZE], f32, [f32; STATE_SIZE], bool)>,
) -> (Vec<[f32; STATE_SIZE]>, Vec<[f32; ACTION_SIZE]>, Vec<f32>, Vec<[f32; STATE_SIZE]>, Vec<bool>) {
    let mut states = Vec::with_capacity(vec.len());
    let mut actions = Vec::with_capacity(vec.len());
    let mut rewards = Vec::with_capacity(vec.len());
    let mut next_states = Vec::with_capacity(vec.len());
    let mut dones = Vec::with_capacity(vec.len());

    for (state, action, reward, next_state, done) in vec {
        states.push(state);
        actions.push(action);
        rewards.push(reward);
        next_states.push(next_state);
        dones.push(done);
    }

    (states, actions, rewards, next_states, dones)
}

pub struct DQNAgentTrainer<
    S,
    const STATE_SIZE: usize,
    const ACTION_SIZE: usize,
    const INNER_SIZE: usize,
    const BUFFER_SIZE: usize,
> where
    S: State + Into<[f32; STATE_SIZE]>,
    S::A: Into<[f32; ACTION_SIZE]>,
    S::A: From<[f32; ACTION_SIZE]>,
{
    // values future rewards
    gamma: f32,
    q_network: QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE>,
    target_q_net: QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE>,
    sgd: Sgd<QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE>, f32, Cpu>,
    dev: Cpu,
    phantom: std::marker::PhantomData<S>,
    replay_buffer: ReplayBuffer<STATE_SIZE, ACTION_SIZE, BUFFER_SIZE>,
    update_counter: usize,
}

impl<S, const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize, const BUFFER_SIZE: usize>
    DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE, BUFFER_SIZE>
where
    S: State + Into<[f32; STATE_SIZE]>,
    S::A: Into<[f32; ACTION_SIZE]>,
    S::A: From<[f32; ACTION_SIZE]>,
{
    /// Creates a new `DQNAgentTrainer` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `gamma` - The discount factor for future rewards.
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new `DQNAgentTrainer` with the given parameters.
    ///
    pub fn new(
        gamma: f32,
        learning_rate: f32,
    ) -> DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE, BUFFER_SIZE> {
        let dev = AutoDevice::default();

        // initialize model
        let q_net = dev.build_module::<QNetwork<STATE_SIZE, ACTION_SIZE, INNER_SIZE>, f32>();
        let target_q_net = q_net.clone();

        // initialize optimizer
        let sgd = Sgd::new(
            &q_net,
            SgdConfig {
                lr: learning_rate as f64,
                momentum: Some(Momentum::Nesterov(0.9)),
                weight_decay: None,
            },
        );

        DQNAgentTrainer {
            gamma,
            q_network: q_net,
            target_q_net,
            sgd,
            dev,
            replay_buffer: ReplayBuffer::new(),
            phantom: std::marker::PhantomData,
            update_counter: 0,
        }
    }

    /// Fetches the learned value for the given `Action` in the given `State`, or `None` if no
    /// value was learned.
    pub fn expected_value(&self, state: &S) -> [f32; ACTION_SIZE] {
        let state_: [f32; STATE_SIZE] = (state.clone()).into();
        let states: Tensor<Rank1<STATE_SIZE>, f32, _> =
            self.dev.tensor(state_).normalize::<Axis<0>>(0.001);
        let actions = self.target_q_net.forward(states).nans_to(0f32);
        actions.array()
    }

    /// Returns a clone of the entire learned state to be saved or used elsewhere.
    pub fn export_learned_values(&self) -> QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE> {
        self.learned_values().clone()
    }

    // Returns a reference to the learned state.
    pub fn learned_values(&self) -> &QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE> {
        &self.q_network
    }

    /// Imports a model, completely replacing any learned progress
    pub fn import_model(&mut self, model: QNetworkDevice<STATE_SIZE, ACTION_SIZE, INNER_SIZE>) {
        self.q_network.clone_from(&model);
        self.target_q_net.clone_from(&self.q_network);
    }

    /// Returns the best action for the given `State`, or `None` if no values were learned.
    pub fn best_action(&self, state: &S) -> Option<S::A> {
        let target = self.expected_value(state);

        Some(target.into())
    }

    #[allow(clippy::boxed_local)]
    pub fn train_dqn(
        &mut self,
        states: Box<[[f32; STATE_SIZE]; BATCH]>,
        actions: [[f32; ACTION_SIZE]; BATCH],
        next_states: Box<[[f32; STATE_SIZE]; BATCH]>,
        rewards: [f32; BATCH],
        dones: [bool; BATCH],
    ) {
        // self.target_q_net.clone_from(&self.q_network);
        // let mut grads = self.q_network.alloc_grads();

        // let dones: Tensor<Rank1<BATCH>, f32, _> =
        //     self.dev.tensor(dones.map(|d| if d { 1f32 } else { 0f32 }));
        // let rewards = self.dev.tensor(rewards);

        // // Convert to tensors and normalize the states for better training
        // let states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> =
        //     self.dev.tensor(*states).normalize::<Axis<1>>(0.001);

        // // Convert actions to tensors and get the max action for each batch
        // let actions: Tensor<Rank1<BATCH>, usize, _> = self.dev.tensor(actions.map(|a| {
        //     let mut max_idx = 0;
        //     let mut max_val = 0f32;
        //     for (i, v) in a.iter().enumerate() {
        //         if *v > max_val {
        //             max_val = *v;
        //             max_idx = i;
        //         }
        //     }
        //     max_idx
        // }));

        // // Convert to tensors and normalize the states for better training
        // let next_states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> =
        //     self.dev.tensor(*next_states).normalize::<Axis<1>>(0.001);

        // // Compute the estimated Q-value for the action
        // for _step in 0..20 {
        //     let q_values = self.q_network.forward(states.trace(grads));

        //     let action_qs = q_values.select(actions.clone());

        //     // targ_q = R + discount * max(Q(S'))
        //     // curr_q = Q(S)[A]
        //     // loss = huber(curr_q, targ_q, 1)
        //     let next_q_values = self.target_q_net.forward(next_states.clone());
        //     let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
        //     let target_q = (max_next_q * (-dones.clone() + 1.0)) * self.gamma + rewards.clone();

        //     let loss = huber_loss(action_qs, target_q, 1.0);

        //     grads = loss.backward();

        //     // update weights with optimizer
        //     self.sgd
        //         .update(&mut self.q_network, &grads)
        //         .expect("Unused params");
        //     self.q_network.zero_grads(&mut grads);
        // }
        // self.target_q_net.clone_from(&self.q_network);
        let dones: Tensor<Rank1<BATCH>, f32, _> = self.dev.tensor(dones.map(|d| if d { 1f32 } else { 0f32 }));
        let rewards = self.dev.tensor(rewards);

        let states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> = self.dev.tensor(*states).normalize::<Axis<1>>(0.01);
        let actions: Tensor<Rank1<BATCH>, usize, _> = self.dev.tensor(actions.map(|a| a.iter().position(|&x| x == 1.0).unwrap()));

        let next_states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> = self.dev.tensor(*next_states).normalize::<Axis<1>>(0.01);

        for _step in 0..20 {
            let q_values = self.q_network.forward(states.trace(self.q_network.alloc_grads()));
            let action_qs = q_values.select(actions.clone());

            let next_q_values = self.target_q_net.forward(next_states.clone());
            let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
            let target_q = (max_next_q * (-dones.clone() + 1.0)) * self.gamma + rewards.clone();

            let loss = huber_loss(action_qs, target_q, 1.0);
            let mut grads = loss.backward();

            self.sgd
                .update(&mut self.q_network, &grads)
                .expect("Unused params");
            self.q_network.zero_grads(&mut grads);
        }

        self.update_counter += 1;
        // ターゲットネットワークの更新頻度をチェック
        if self.update_counter % TARGET_UPDATE_FREQ == 0 {
            self.target_q_net.clone_from(&self.q_network);
        }
    }

    /// Trains this [DQNAgentTrainer] using the given [ExplorationStrategy] and
    /// [Agent] until the [TerminationStrategy] decides to stop.
    pub fn train(
        &mut self,
        agent: &mut dyn Agent<S>,
        termination_strategy: &mut dyn TerminationStrategy<S>,
        exploration_strategy: &mut dyn ExplorationStrategy<S>,
        reward_history: &mut Vec<f32>,
    ) {
        loop {
            // Initialize batch
            // let mut states: Box<[[f32; STATE_SIZE]; BATCH]> = {
            //     let b = vec![0.0; STATE_SIZE].into_boxed_slice();
            //     let big = unsafe { Box::from_raw(Box::into_raw(b) as *mut [f32; STATE_SIZE]) };

            //     let b = vec![*big; BATCH].into_boxed_slice();
            //     unsafe { Box::from_raw(Box::into_raw(b) as *mut [[f32; STATE_SIZE]; BATCH]) }
            // };
            // let mut actions: [[f32; ACTION_SIZE]; BATCH] = [[0.0; ACTION_SIZE]; BATCH];
            // let mut next_states: Box<[[f32; STATE_SIZE]; BATCH]> = {
            //     let b = vec![0.0; STATE_SIZE].into_boxed_slice();
            //     let big = unsafe { Box::from_raw(Box::into_raw(b) as *mut [f32; STATE_SIZE]) };

            //     let b = vec![*big; BATCH].into_boxed_slice();
            //     unsafe { Box::from_raw(Box::into_raw(b) as *mut [[f32; STATE_SIZE]; BATCH]) }
            // };
            // let mut rewards: [f32; BATCH] = [0.0; BATCH];
            // let mut dones = [false; BATCH];

            let mut round_reward = 0.0;
            let mut s_t_next = agent.current_state();

            for _ in 0..BATCH {
                let s_t = agent.current_state().clone();
                let action = exploration_strategy.pick_action(agent);

                // current action value
                s_t_next = agent.current_state();
                let r_t_next = s_t_next.reward();

                self.replay_buffer.add(
                    s_t.into(), 
                    action.into(), 
                    r_t_next as f32, 
                    <S as Clone>::clone(&(*s_t_next)).into(), 
                    termination_strategy.should_stop(s_t_next)
                );
                // states[i] = s_t.into();
                // actions[i] = action.into();
                // next_states[i] = (*s_t_next).clone().into();
                // rewards[i] = r_t_next as f32;
                // round_reward += rewards[i];
                round_reward += r_t_next as f32;
                if termination_strategy.should_stop(s_t_next) {
                    // dones[i] = true;
                    break;
                }
            }
            reward_history.push(round_reward);

            if self.replay_buffer.full {
                let batch = self.replay_buffer.sample(BATCH);
                let (states,actions, rewards, next_states, dones) = unzip_n(batch);
                self.train_dqn(Box::new(states.try_into().unwrap()), actions.try_into().unwrap(), Box::new(next_states.try_into().unwrap()), rewards.try_into().unwrap(), dones.try_into().unwrap());
            }
            // self.train_dqn(states, actions, next_states, rewards, dones);
            if termination_strategy.should_stop(s_t_next) {
                break;
            }
        }
    }
}

impl<S, const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize, const BUFFER_SIZE: usize> Default
    for DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE, BUFFER_SIZE>
where
    S: State + Into<[f32; STATE_SIZE]>,
    S::A: Into<[f32; ACTION_SIZE]>,
    S::A: From<[f32; ACTION_SIZE]>,
{
    fn default() -> Self {
        Self::new(0.99, 0.01)
    }
}