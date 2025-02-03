//! 繰り返し学習させるアプリ

use std::{
    fmt::{Display, Formatter, Result},
    process::{Child, Command},
    thread,
    time::Duration,
};

use clap::{Parser, ValueEnum};
use engarde_client::print;

const FINAL_LOOP_COUNT: usize = 20;
const LOOP_COUNT: usize = 20;
const MAX_ROUND: u32 = 100;

#[derive(ValueEnum, Clone, Debug, Copy)]
enum Client {
    Train,
    Eval,
    Random,
    RandomForward,
    Algorithm,
    Aggressive,
}

impl Client {
    fn execute(&self) -> Child {
        match self {
            Self::Train => Command::new(".\\dqn.exe")
                .arg("-m")
                .arg("train")
                .spawn()
                .expect("dqn.exe起動失敗"),
            Self::Eval => Command::new(".\\dqn.exe")
                .arg("-m")
                .arg("eval")
                .spawn()
                .expect("dqn.exe起動失敗"),
            Self::Random => Command::new(".\\random.exe")
                .spawn()
                .expect("random.exe起動失敗"),
            Self::RandomForward => Command::new(".\\random_forward.exe")
                .spawn()
                .expect("random_forward.exe起動失敗"),
            Self::Algorithm => Command::new(".\\using_algorithm.exe")
                .spawn()
                .expect("using_algorithm.exe起動失敗"),
            Self::Aggressive => Command::new(".\\aggressive.exe")
                .spawn()
                .expect("aggressive.exe起動失敗"),
        }
    }
}

impl Display for Client {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let s = match self {
            Self::Train => "train",
            Self::Eval => "eval",
            Self::Random => "random",
            Self::RandomForward => "random_forward",
            Self::Algorithm => "algorithm",
            Self::Aggressive => "aggressive",
        };
        s.fmt(f)
    }
}

#[derive(ValueEnum, Clone, Debug)]
enum LearningMode {
    QLearning,
    Dqn,
}

impl Display for LearningMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let s = match self {
            LearningMode::QLearning => "q-learning",
            LearningMode::Dqn => "dqn",
        };
        s.fmt(f)
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, short, default_value_t = LearningMode::Dqn)]
    learning_mode: LearningMode,
    #[arg(long, short, default_value_t = FINAL_LOOP_COUNT)]
    final_loop: usize,
    #[arg(long, short = 'c', default_value_t = LOOP_COUNT)]
    loop_count: usize,
    #[arg(long, short, default_value_t = MAX_SCORE)]
    max_score: u32,
}

fn q_learning_loop(final_loop: usize, loop_count: usize, max_score: u32) {
    for _ in 0..final_loop {
        let mut client0 = Command::new(".\\q-learning.exe")
            .arg("-m")
            .arg("train")
            .arg("-i")
            .arg(0.to_string())
            .arg("-l")
            .arg(loop_count.to_string())
            .spawn()
            .expect("q-learning.exe起動失敗");
        let mut client1 = Command::new(".\\q-learning.exe")
            .arg("-m")
            .arg("train")
            .arg("-i")
            .arg(1.to_string())
            .arg("-l")
            .arg(loop_count.to_string())
            .spawn()
            .expect("q-learning.exe起動失敗");
        for _ in 0..loop_count {
            let mut server = Command::new(".\\engarde_server.exe")
                .arg(max_score.to_string())
                .spawn()
                .expect("engarde_server.exe起動失敗");
            server.wait().expect("engarde_serverクラッシュ");
        }
        client0.wait().expect("q-learning.exeクラッシュ");
        client1.wait().expect("q-learning.exeクラッシュ");
    }
}

fn dqn_loop(final_loop: usize, loop_count: usize, max_score: u32) {
    for i in 0..final_loop * loop_count {
        let mut server = Command::new(".\\engarde_server.exe")
            .arg(max_score.to_string())
            .spawn()
            .expect("engarde_server.exe起動失敗");
        let mut client0 = client0.execute();
        thread::sleep(Duration::from_millis(50));
        let mut client1 = client1.execute();
        server.wait().expect("engarde_serverクラッシュ");
        client0.wait().expect("dqn.exeクラッシュ");
        client1.wait().expect("dqn.exeクラッシュ");
        print(format!("{i}")).expect("出力に失敗");
    }
}

fn main() {
    let args = Args::parse();
    client_loop(args.player0, args.player1, args.loop_count, args.max_round);
}
