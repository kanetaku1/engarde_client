// source: https://raw.githubusercontent.com/coreylowman/dfdx/main/examples/rl-dqn.rs
use dfdx::{
    nn,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};

use crate::{
    mdp::{Agent, State},
    strategy::{explore::ExplorationStrategy, terminate::TerminationStrategy},
};

const BATCH: usize = 1024;

#[derive(Default, Debug, Clone, Copy)]
pub struct Mish;

impl ZeroSizedModule for Mish {}
impl NonMutableModule for Mish {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Mish {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        let mut cloned = input.device().zeros_like(&input).retaped::<T>();
        cloned.copy_from(&input.as_vec());
        Ok(cloned
            * (input.device().ones_like(&input).retaped::<T>() + input.try_exp()?)
                .try_ln()?
                .try_tanh()?)
    }
}

type QNetwork<const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize> = (
    (Linear<STATE_SIZE, INNER_SIZE>, Mish),
    (Linear<INNER_SIZE, INNER_SIZE>, Mish),
    (Linear<INNER_SIZE, INNER_SIZE>, Mish),
    Linear<INNER_SIZE, ACTION_SIZE>,
);

type QNetworkDevice<const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize> = (
    (nn::modules::Linear<STATE_SIZE, INNER_SIZE, f32, Cpu>, Mish),
    (nn::modules::Linear<INNER_SIZE, INNER_SIZE, f32, Cpu>, Mish),
    (nn::modules::Linear<INNER_SIZE, INNER_SIZE, f32, Cpu>, Mish),
    nn::modules::Linear<INNER_SIZE, ACTION_SIZE, f32, Cpu>,
);

/// An `DQNAgentTrainer` can be trained for using a certain [Agent](mdp/trait.Agent.html). After
/// training, the `DQNAgentTrainer` contains learned knowledge about the process, and can be queried
/// for this. For example, you can ask the `DQNAgentTrainer` the expected values of all possible
/// actions in a given state.
///
/// The code is partially taken from https://github.com/coreylowman/dfdx/blob/main/examples/rl-dqn.rs.
///
pub struct DQNAgentTrainer<
    S,
    const STATE_SIZE: usize,
    const ACTION_SIZE: usize,
    const INNER_SIZE: usize,
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
}

impl<S, const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize>
    DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE>
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
    ) -> DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE> {
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
            phantom: std::marker::PhantomData,
        }
    }

    /// Fetches the learned value for the given `Action` in the given `State`, or `None` if no
    /// value was learned.
    pub fn expected_value(&self, state: &S) -> [f32; ACTION_SIZE] {
        let state_: [f32; STATE_SIZE] = (state.clone()).into();
        // println!("State: {:?}", state_);
        let states: Tensor<Rank1<STATE_SIZE>, f32, _> = self.dev.tensor(state_).normalize::<Axis<0>>(0.001);
        // println!("Normalized State: {:?}", states);
        let actions = self.target_q_net.forward(states).nans_to(0f32);
        // println!("Actions: {:?}", actions);
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
    // pub fn best_action(&self, state: &S) -> Option<S::A> {
        // let target = self.expected_value(state);

        // // // 最大のQ値を持つ行動を選択
        // // let best_action_index = target
        // //     .iter()
        // //     .enumerate()
        // //     .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
        // //     .map(|(index, _)| index)
        // //     .unwrap();

        // // best_action_index
        // Some(target.into())
    pub fn best_action(&self, state: &S, valid_actions: &[S::A]) -> Option<S::A> {
        let target = self.expected_value(state);
        
        // 有効なアクションのQ値のみを考慮
        let best_action = valid_actions.iter()
            .max_by(|&a1, &a2| {
                let idx1 = self.action_to_index(a1);
                let idx2 = self.action_to_index(a2);
                target[idx1].partial_cmp(&target[idx2]).unwrap()
            })
            .cloned();
        best_action
    }

    fn action_to_index(&self, action: &S::A) -> usize {
        let action_array: [f32; ACTION_SIZE] = action.clone().into();
        action_array.iter().position(|&x| x == 1.0).unwrap()
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
        self.target_q_net.clone_from(&self.q_network);
        let mut grads = self.q_network.alloc_grads();

        // let dones: Tensor<Rank1<BATCH>, f32, _> =
        //     self.dev.tensor(dones.map(|d| if d { 1f32 } else { 0f32 }));
        // let scaled_rewards = rewards.map(|r| r * 100.0);  // 報酬を大きくスケーリング
        // let rewards = self.dev.tensor(scaled_rewards);
        // // let rewards = self.dev.tensor(rewards);

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
        let scaled_rewards = rewards.map(|r| r * 100.0);  // 報酬を大きくスケーリング
        let rewards = self.dev.tensor(scaled_rewards);
        // let rewards = self.dev.tensor(rewards);

        // Convert to tensors and normalize the states for better training
        let states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> =
            self.dev.tensor(*states).normalize::<Axis<1>>(0.001);

        // Convert actions to tensors and get the max action for each batch
        let actions: Tensor<Rank1<BATCH>, usize, _> = self.dev.tensor(actions.map(|a| {
            let mut max_idx = 0;
            let mut max_val = 0f32;
            for (i, v) in a.iter().enumerate() {
                if *v > max_val {
                    max_val = *v;
                    max_idx = i;
                }
            }
            max_idx
        }));

        // Convert to tensors and normalize the states for better training
        let next_states: Tensor<Rank2<BATCH, STATE_SIZE>, f32, _> =
            self.dev.tensor(*next_states).normalize::<Axis<1>>(0.001);

        // Compute the estimated Q-value for the action
        for _step in 0..20 {
            let q_values = self.q_network.forward(states.trace(grads));

            let action_qs = q_values.select(actions.clone());

            // targ_q = R + discount * max(Q(S'))
            // curr_q = Q(S)[A]
            // loss = huber(curr_q, targ_q, 1)
            let next_q_values = self.target_q_net.forward(next_states.clone());
            let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
            let target_q = (max_next_q * (-dones.clone() + 1.0)) * self.gamma + rewards.clone();

            let loss = huber_loss(action_qs, target_q, 1.0);

            grads = loss.backward();

            // update weights with optimizer
            self.sgd
                .update(&mut self.q_network, &grads)
                .expect("Unused params");
            self.q_network.zero_grads(&mut grads);
        }
        self.target_q_net.clone_from(&self.q_network);
    }

    /// Trains this [DQNAgentTrainer] using the given [ExplorationStrategy] and
    /// [Agent] until the [TerminationStrategy] decides to stop.
    pub fn train(
        &mut self,
        agent: &mut dyn Agent<S>,
        termination_strategy: &mut dyn TerminationStrategy<S>,
        exploration_strategy: &dyn ExplorationStrategy<S>,
    ) {
        loop {
            // Initialize batch
            let mut states: Box<[[f32; STATE_SIZE]; BATCH]> = {
                let b = vec![0.0; STATE_SIZE].into_boxed_slice();
                let big = unsafe { Box::from_raw(Box::into_raw(b) as *mut [f32; STATE_SIZE]) };

                let b = vec![*big; BATCH].into_boxed_slice();
                unsafe { Box::from_raw(Box::into_raw(b) as *mut [[f32; STATE_SIZE]; BATCH]) }
            };
            let mut actions: [[f32; ACTION_SIZE]; BATCH] = [[0.0; ACTION_SIZE]; BATCH];
            let mut next_states: Box<[[f32; STATE_SIZE]; BATCH]> = {
                let b = vec![0.0; STATE_SIZE].into_boxed_slice();
                let big = unsafe { Box::from_raw(Box::into_raw(b) as *mut [f32; STATE_SIZE]) };

                let b = vec![*big; BATCH].into_boxed_slice();
                unsafe { Box::from_raw(Box::into_raw(b) as *mut [[f32; STATE_SIZE]; BATCH]) }
            };
            let mut rewards: [f32; BATCH] = [0.0; BATCH];
            let mut dones = [false; BATCH];

            let mut s_t_next = agent.current_state();

            for i in 0..BATCH {
                let s_t = agent.current_state().clone();
                let action = exploration_strategy.pick_action(agent);

                // current action value
                s_t_next = agent.current_state();
                let r_t_next = s_t_next.reward();

                states[i] = s_t.into();
                actions[i] = action.into();
                next_states[i] = (*s_t_next).clone().into();
                rewards[i] = r_t_next as f32;

                if termination_strategy.should_stop(s_t_next) {
                    dones[i] = true;
                     // エピソード終了時の報酬を設定
                     let final_reward = s_t_next.final_score_reward();
                     for j in 0..=i {
                         rewards[j] += final_reward as f32 * self.gamma.powi((i - j) as i32);
                     }
                    break;
                }
            }
            assert!(dones.into_iter().any(|x| x), "満タン!");
            // train the network
            self.train_dqn(states, actions, next_states, rewards, dones);

            // terminate if the agent is done
            if termination_strategy.should_stop(s_t_next) {
                break;
            }
        }
    }
}

impl<S, const STATE_SIZE: usize, const ACTION_SIZE: usize, const INNER_SIZE: usize> Default
    for DQNAgentTrainer<S, STATE_SIZE, ACTION_SIZE, INNER_SIZE>
where
    S: State + Into<[f32; STATE_SIZE]>,
    S::A: Into<[f32; ACTION_SIZE]>,
    S::A: From<[f32; ACTION_SIZE]>,
{
    fn default() -> Self {
        Self::new(0.99, 1e-3)
    }
}
