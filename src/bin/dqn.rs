//! DQNのAIクライアント

use std::{
    cmp::Ordering,
    fs::{self, create_dir_all},
    io::{self, BufReader, BufWriter},
    net::{SocketAddr, TcpStream},
};

use apply::Also;
use clap::{Parser, ValueEnum};
use dfdx::{
    nn::modules::{Linear, ReLU},
    shapes::Const,
    tensor::{Cpu, NoneTape, Tensor, ZerosTensor},
};
use rand::{thread_rng, Rng};
use rurel::{
    dqn::DQNAgentTrainer,
    mdp::{Agent, State},
    strategy::{
        explore::{ExplorationStrategy, RandomExploration},
        terminate::{SinkStates, TerminationStrategy},
    },
};

use engarde_client::{
    get_id,
    protocol::{BoardInfo, Messages, PlayerName},
    read_stream, send_info,
    states::{MyAgent, MyState},
    Action, CardID, Direction,
};

type DQNAgentTrainerContinuous = DQNAgentTrainer<MyState, 15, 3, 128>;
type WeightInTensor = Tensor<(Const<INNER_CONTINUOUS>, Const<15>), f32, Cpu>;
type BiasInTensor = Tensor<(Const<INNER_CONTINUOUS>,), f32, Cpu>;
type WeightInnerTensor =
    Tensor<(Const<INNER_CONTINUOUS>, Const<INNER_CONTINUOUS>), f32, Cpu, NoneTape>;
type BiasInnerTensor = Tensor<(Const<INNER_CONTINUOUS>,), f32, Cpu>;
type WeightOutTensor = Tensor<(Const<ACTION_SIZE_CONTINUOUS>, Const<INNER_CONTINUOUS>), f32, Cpu>;
type BiasOutTensor = Tensor<(Const<ACTION_SIZE_CONTINUOUS>,), f32, Cpu>;

const INNER_CONTINUOUS: usize = 128;
const ACTION_SIZE_CONTINUOUS: usize = 3;

const DISCOUNT_RATE: f32 = 0.9;
const LEARNING_RATE: f32 = 0.1;

/// ベストに近いアクションを返す
#[allow(dead_code, clippy::too_many_lines)]
fn neary_best_action(state: &MyState, trainer: &DQNAgentTrainerContinuous) -> Option<Action> {
    let best = trainer.best_action(state)?;
    let actions = state.actions();
    if actions.contains(&best) {
        Some(best)
    } else {
        match best {
            Action::Move(movement) => {
                // 前進の場合、前進-攻撃-後退の順に並び替え
                // 後退の場合、後退-攻撃-前進の順に並び替え
                // (クソ長い)
                fn ordering(
                    key_direction: Direction,
                    key_card: CardID,
                    action1: Action,
                    action2: Action,
                ) -> Ordering {
                    match action1 {
                        Action::Move(movement1) => {
                            let card1 = movement1.card();
                            let direction1 = movement1.direction();
                            match direction1 {
                                Direction::Forward => match action2 {
                                    Action::Move(movement2) => {
                                        let card2 = movement2.card();
                                        let direction2 = movement2.direction();
                                        match direction2 {
                                            Direction::Forward => {
                                                let key_card_i32 = i32::from(key_card.denote());
                                                let card1_i32 = i32::from(card1.denote());
                                                let card2_i32 = i32::from(card2.denote());
                                                (card1_i32 - key_card_i32)
                                                    .abs()
                                                    .cmp(&(card2_i32 - key_card_i32).abs())
                                            }
                                            Direction::Back => match key_direction {
                                                Direction::Forward => Ordering::Less,
                                                Direction::Back => Ordering::Greater,
                                            },
                                        }
                                    }
                                    Action::Attack(_) => match key_direction {
                                        Direction::Forward => Ordering::Less,
                                        Direction::Back => Ordering::Greater,
                                    },
                                },
                                Direction::Back => match action2 {
                                    Action::Move(movement2) => {
                                        let card2 = movement2.card();
                                        let direction2 = movement2.direction();
                                        match direction2 {
                                            Direction::Forward => match key_direction {
                                                Direction::Forward => Ordering::Greater,
                                                Direction::Back => Ordering::Less,
                                            },
                                            Direction::Back => {
                                                let key_card_i32 = i32::from(key_card.denote());
                                                let card1_i32 = i32::from(card1.denote());
                                                let card2_i32 = i32::from(card2.denote());
                                                (card1_i32 - key_card_i32)
                                                    .abs()
                                                    .cmp(&(card2_i32 - key_card_i32).abs())
                                            }
                                        }
                                    }
                                    Action::Attack(_) => match key_direction {
                                        Direction::Forward => Ordering::Greater,
                                        Direction::Back => Ordering::Less,
                                    },
                                },
                            }
                        }
                        Action::Attack(_) => match action2 {
                            Action::Move(movement2) => {
                                let direction2 = movement2.direction();
                                match direction2 {
                                    Direction::Forward => match key_direction {
                                        Direction::Forward => Ordering::Greater,
                                        Direction::Back => Ordering::Less,
                                    },
                                    Direction::Back => match key_direction {
                                        Direction::Forward => Ordering::Less,
                                        Direction::Back => Ordering::Greater,
                                    },
                                }
                            }
                            Action::Attack(_) => Ordering::Equal,
                        },
                    }
                }
                let card = movement.card();
                let direction = movement.direction();
                match direction {
                    Direction::Forward => {
                        let mut actions = actions;
                        actions.sort_by(|&action1, &action2| {
                            ordering(Direction::Forward, card, action1, action2)
                        });
                        actions.first().copied()
                    }
                    Direction::Back => {
                        let mut actions = actions;
                        actions.sort_by(|&action1, &action2| {
                            ordering(Direction::Back, card, action1, action2)
                        });
                        actions.first().copied()
                    }
                }
            }
            Action::Attack(_) => {
                let mut actions = actions;
                actions.sort_by_key(|action| match action {
                    Action::Move(movement) => movement.card(),
                    Action::Attack(attack) => attack.card(),
                });
                actions.first().copied()
            }
        }
    }
}

struct EpsilonGreedyContinuous {
    past_exp: DQNAgentTrainerContinuous,
    epsilon: u64,
}

impl EpsilonGreedyContinuous {
    fn new(trainer: DQNAgentTrainerContinuous, start_epsilon: u64) -> Self {
        EpsilonGreedyContinuous {
            past_exp: trainer,
            epsilon: start_epsilon,
        }
    }
}

impl ExplorationStrategy<MyState> for EpsilonGreedyContinuous {
    fn pick_action(&mut self, agent: &mut dyn Agent<MyState>) -> <MyState as State>::A {
        let random = thread_rng().gen::<u64>();
        if random < self.epsilon {
            agent.pick_random_action()
        } else {
            match neary_best_action(agent.current_state(), &self.past_exp) {
                Some(action) => {
                    agent.take_action(&action);
                    action
                }
                None => agent.pick_random_action(),
            }
        }
    }
}

struct BestExplorationDqnContinuous(DQNAgentTrainerContinuous);

impl BestExplorationDqnContinuous {
    fn new(trainer: DQNAgentTrainerContinuous) -> Self {
        BestExplorationDqnContinuous(trainer)
    }
}

impl ExplorationStrategy<MyState> for BestExplorationDqnContinuous {
    fn pick_action(&mut self, agent: &mut dyn Agent<MyState>) -> <MyState as State>::A {
        match neary_best_action(agent.current_state(), &self.0) {
            Some(action) => {
                agent.take_action(&action);
                action
            }
            None => agent.pick_random_action(),
        }
    }
}

struct NNFileNames {
    weight_in: String,
    bias_in: String,
    weight1: String,
    bias1: String,
    weight2: String,
    bias2: String,
    weight_out: String,
    bias_out: String,
}

fn files_name(id: u8) -> NNFileNames {
    NNFileNames {
        weight_in: format!("learned_dqn/{id}/weight_in.npy"),
        bias_in: format!("learned_dqn/{id}/bias_in.npy"),
        weight1: format!("learned_dqn/{id}/weight1.npy"),
        bias1: format!("learned_dqn/{id}/bias1.npy"),
        weight2: format!("learned_dqn/{id}/weight2.npy"),
        bias2: format!("learned_dqn/{id}/bias2.npy"),
        weight_out: format!("learned_dqn/{id}/weight_out.npy"),
        bias_out: format!("learned_dqn/{id}/bias_out.npy"),
    }
}

fn dqn_train() -> io::Result<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], 12052));
    let stream = loop {
        if let Ok(stream) = TcpStream::connect(addr) {
            break stream;
        }
    };
    let (mut bufreader, mut bufwriter) =
        (BufReader::new(stream.try_clone()?), BufWriter::new(stream));
    let id = get_id(&mut bufreader)?;
    let player_name = PlayerName::new("dqnai".to_string());
    send_info(&mut bufwriter, &player_name)?;
    let _ = read_stream(&mut bufreader)?;
    // ここは、最初に自分が持ってる手札を取得するために、AIの行動じゃなしに情報を得なならん
    let mut board_info_init = BoardInfo::new();

    let hand_info = loop {
        match Messages::parse(&read_stream(&mut bufreader)?) {
            Ok(Messages::BoardInfo(board_info)) => {
                board_info_init = board_info;
            }
            Ok(Messages::HandInfo(hand_info)) => {
                break hand_info;
            }
            Ok(_) | Err(_) => {}
        }
    };
    let hand_vec = hand_info.to_vec().also(|hand_vec| hand_vec.sort());
    // AI用エージェント作成
    let mut agent = MyAgent::new(
        id,
        hand_vec,
        board_info_init.p0_position(),
        board_info_init.p1_position(),
        bufreader,
        bufwriter,
    );

    // let mut trainer = DQNAgentTrainerDiscreate::new(DISCOUNT_RATE, LEARNING_RATE);
    let mut trainer = DQNAgentTrainerContinuous::new(DISCOUNT_RATE, LEARNING_RATE);
    let past_exp = {
        let cpu = Cpu::default();
        let mut weight_in: WeightInTensor = cpu.zeros();
        let mut bias_in: BiasInTensor = cpu.zeros();
        let mut weight1: WeightInnerTensor = cpu.zeros();
        let mut bias1: BiasInnerTensor = cpu.zeros();
        let mut weight2: WeightInnerTensor = cpu.zeros();
        let mut bias2: BiasInnerTensor = cpu.zeros();
        let mut weight_out: WeightOutTensor = cpu.zeros();
        let mut bias_out: BiasOutTensor = cpu.zeros();
        let files = files_name(id.denote());
        (|| {
            weight_in.load_from_npy(files.weight_in).ok()?;
            bias_in.load_from_npy(files.bias_in).ok()?;
            weight1.load_from_npy(files.weight1).ok()?;
            bias1.load_from_npy(files.bias1).ok()?;
            weight2.load_from_npy(files.weight2).ok()?;
            bias2.load_from_npy(files.bias2).ok()?;
            weight_out.load_from_npy(files.weight_out).ok()?;
            bias_out.load_from_npy(files.bias_out).ok()?;
            Some(())
        })()
        .map_or(trainer.export_learned_values(), |()| {
            (
                (
                    Linear {
                        weight: weight_in,
                        bias: bias_in,
                    },
                    ReLU,
                ),
                (
                    Linear {
                        weight: weight1,
                        bias: bias1,
                    },
                    ReLU,
                ),
                (
                    Linear {
                        weight: weight2,
                        bias: bias2,
                    },
                    ReLU,
                ),
                Linear {
                    weight: weight_out,
                    bias: bias_out,
                },
            )
        })
    };
    trainer.import_model(past_exp.clone());
    let mut trainer2 = DQNAgentTrainer::new(DISCOUNT_RATE, LEARNING_RATE);
    trainer2.import_model(past_exp);
    let epsilon = fs::read_to_string(format!("learned_dqn/{}/epsilon.txt", id.denote()))
        .map(|eps_str| eps_str.parse::<u64>().expect("εが適切なu64値でない"))
        .unwrap_or(u64::MAX);
    let epsilon = (epsilon - (epsilon / 200)).max(u64::MAX / 20);
    let mut epsilon_greedy_exploration = EpsilonGreedyContinuous::new(trainer2, epsilon);
    trainer.train(
        &mut agent,
        &mut SinkStates {},
        &mut epsilon_greedy_exploration,
    );
    {
        let learned_values = trainer.export_learned_values();
        let linear_in = learned_values.0 .0;
        let weight_in = linear_in.weight;
        let bias_in = linear_in.bias;
        let linear1 = learned_values.1 .0;
        let weight1 = linear1.weight;
        let bias1 = linear1.bias;
        let linear2 = learned_values.2 .0;
        let weight2 = linear2.weight;
        let bias2 = linear2.bias;
        let linear_out = learned_values.3;
        let weight_out = linear_out.weight;
        let bias_out = linear_out.bias;
        let files = files_name(id.denote());
        let _ = create_dir_all(format!("learned_dqn/{}", id.denote()));
        weight_in.save_to_npy(files.weight_in)?;
        bias_in.save_to_npy(files.bias_in)?;
        weight1.save_to_npy(files.weight1)?;
        bias1.save_to_npy(files.bias1)?;
        weight2.save_to_npy(files.weight2)?;
        bias2.save_to_npy(files.bias2)?;
        weight_out.save_to_npy(files.weight_out)?;
        bias_out.save_to_npy(files.bias_out)?;
        fs::write(
            format!("learned_dqn/{}/epsilon.txt", id.denote()),
            epsilon_greedy_exploration.epsilon.to_string(),
        )?;
    }
    Ok(())
}

fn evaluation(
    agent: &mut MyAgent,
    termination_strategy: &mut dyn TerminationStrategy<MyState>,
    best_exploration_strategy: &mut BestExplorationDqnContinuous,
) {
    loop {
        best_exploration_strategy.pick_action(agent);
        if termination_strategy.should_stop(agent.current_state()) {
            break;
        }
    }
}

fn dqn_eval() -> io::Result<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], 12052));
    let stream = loop {
        if let Ok(stream) = TcpStream::connect(addr) {
            break stream;
        }
    };
    let (mut bufreader, mut bufwriter) =
        (BufReader::new(stream.try_clone()?), BufWriter::new(stream));
    let id = get_id(&mut bufreader)?;
    let player_name = PlayerName::new("dqnai".to_string());
    send_info(&mut bufwriter, &player_name)?;
    let _ = read_stream(&mut bufreader)?;
    // ここは、最初に自分が持ってる手札を取得するために、AIの行動じゃなしに情報を得なならん
    let mut board_info_init = BoardInfo::new();

    let hand_info = loop {
        match Messages::parse(&read_stream(&mut bufreader)?) {
            Ok(Messages::BoardInfo(board_info)) => {
                board_info_init = board_info;
            }
            Ok(Messages::HandInfo(hand_info)) => {
                break hand_info;
            }
            Ok(_) | Err(_) => {}
        }
    };
    let hand_vec = hand_info.to_vec().also(|hand_vec| hand_vec.sort());
    // AI用エージェント作成
    let mut agent = MyAgent::new(
        id,
        hand_vec,
        board_info_init.p0_position(),
        board_info_init.p1_position(),
        bufreader,
        bufwriter,
    );

    let mut trainer = DQNAgentTrainerContinuous::new(DISCOUNT_RATE, LEARNING_RATE);
    let past_exp = {
        let cpu = Cpu::default();
        let mut weight_in: WeightInTensor = cpu.zeros();
        let mut bias_in: BiasInTensor = cpu.zeros();
        let mut weight1: WeightInnerTensor = cpu.zeros();
        let mut bias1: BiasInnerTensor = cpu.zeros();
        let mut weight2: WeightInnerTensor = cpu.zeros();
        let mut bias2: BiasInnerTensor = cpu.zeros();
        let mut weight_out: WeightOutTensor = cpu.zeros();
        let mut bias_out: BiasOutTensor = cpu.zeros();
        let files = files_name(id.denote());
        (|| {
            weight_in.load_from_npy(files.weight_in).ok()?;
            bias_in.load_from_npy(files.bias_in).ok()?;
            weight1.load_from_npy(files.weight1).ok()?;
            bias1.load_from_npy(files.bias1).ok()?;
            weight2.load_from_npy(files.weight2).ok()?;
            bias2.load_from_npy(files.bias2).ok()?;
            weight_out.load_from_npy(files.weight_out).ok()?;
            bias_out.load_from_npy(files.bias_out).ok()?;
            Some(())
        })()
        .map_or(trainer.export_learned_values(), |()| {
            (
                (
                    Linear {
                        weight: weight_in,
                        bias: bias_in,
                    },
                    ReLU,
                ),
                (
                    Linear {
                        weight: weight1,
                        bias: bias1,
                    },
                    ReLU,
                ),
                (
                    Linear {
                        weight: weight2,
                        bias: bias2,
                    },
                    ReLU,
                ),
                Linear {
                    weight: weight_out,
                    bias: bias_out,
                },
            )
        })
    };
    trainer.import_model(past_exp.clone());
    evaluation(
        &mut agent,
        &mut SinkStates {},
        &mut BestExplorationDqnContinuous::new(trainer),
    );
    Ok(())
}

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Train,
    Eval,
}

#[derive(Parser, Debug)]
struct Arguments {
    #[arg(long, short)]
    mode: Mode,
}

fn main() -> io::Result<()> {
    let args = Arguments::parse();
    match args.mode {
        Mode::Train => dqn_train(),
        Mode::Eval => dqn_eval(),
    }
}
