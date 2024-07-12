//! AI用の状態・エージェント等々
//! 正直ごちゃごちゃ入れすぎているから良くない　双依存になってる

use std::{
    collections::HashSet,
    hash::RandomState,
    io::{self, BufReader, BufWriter},
    net::TcpStream,
    ops::Mul,
};

use apply::Also;
use num_rational::Ratio;
use num_traits::{ToPrimitive, Zero};
use rurel::mdp::{Agent, State};

use crate::{
    algorithm::{card_map_from_hands, safe_possibility, ProbabilityTable},
    print,
    protocol::{Evaluation, Messages, PlayAttack, PlayMovement, PlayerID},
    read_stream, send_info, Action, Attack, CardID, Direction, Maisuu, Movement, UsedCards,
};

/// Stateは、結果状態だけからその評価と次できる行動のリストを与える。
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct MyState {
    my_id: PlayerID,
    hands: Vec<CardID>,
    used: UsedCards,
    p0_score: u32,
    p1_score: u32,
    p0_position: u8,
    p1_position: u8,
    game_end: bool,
    prev_action: Option<Action>,
    round_winner: i8,
}

impl MyState {
    /// 手札を返します。
    pub fn hands(&self) -> &[CardID] {
        &self.hands
    }

    /// 自分のプレイヤーIDを返します。
    pub fn my_id(&self) -> PlayerID {
        self.my_id
    }

    /// `UsedCards`を返します。
    pub fn used_cards(&self) -> UsedCards {
        self.used
    }

    /// プレイヤー0のスコアを返します。
    pub fn p0_score(&self) -> u32 {
        self.p0_score
    }

    /// プレイヤー1のスコアを返します。
    pub fn p1_score(&self) -> u32 {
        self.p1_score
    }

    /// プレイヤー0の位置を返します。
    pub fn p0_position(&self) -> u8 {
        self.p0_position
    }

    /// プレイヤー1の位置を返します。
    pub fn p1_position(&self) -> u8 {
        self.p1_position
    }

    /// ゲームが終了したかどうかを返します。
    pub fn game_end(&self) -> bool {
        self.game_end
    }

    /// `MyState`を生成します。
    // いやごめんてclippy
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        my_id: PlayerID,
        hands: Vec<CardID>,
        used: UsedCards,
        p0_score: u32,
        p1_score: u32,
        p0_position: u8,
        p1_position: u8,
        game_end: bool,
        prev_action: Option<Action>,
        round_winner: i8,
    ) -> Self {
        Self {
            my_id,
            hands,
            used,
            p0_score,
            p1_score,
            p0_position,
            p1_position,
            game_end,
            prev_action,
            round_winner,
        }
    }

    fn my_score(&self) -> u32 {
        match self.my_id {
            PlayerID::Zero => self.p0_score,
            PlayerID::One => self.p1_score,
        }
    }

    fn enemy_score(&self) -> u32 {
        match self.my_id {
            PlayerID::Zero => self.p1_score,
            PlayerID::One => self.p0_score,
        }
    }

    fn calc_dist(&self) -> u8 {
        self.p1_position - self.p0_position
    }

    fn calc_safe_reward(&self) -> f64 {
        let actions = self.actions();
        let card_map = card_map_from_hands(&self.hands).expect("安心して");
        actions
            .iter()
            .map(|&action| {
                safe_possibility(
                    self.calc_dist(),
                    self.used_cards().to_restcards(card_map),
                    self.hands(),
                    &ProbabilityTable::new(&self.used_cards().to_restcards(card_map)),
                    action,
                )
            })
            .sum::<Option<Ratio<u64>>>()
            .unwrap_or(Ratio::<u64>::zero())
            .to_f64()
            .expect("なんで")
            .mul(10.0)
    }

    #[allow(clippy::float_arithmetic)]
    fn calc_score_reward(&self) -> f64 {
        (f64::from(self.my_score())) - (f64::from(self.enemy_score()))
    }

    fn distance_from_center(&self) -> i8 {
        match self.my_id {
            PlayerID::Zero => 12 - i8::try_from(self.p0_position).expect("i8の表現範囲外"),
            PlayerID::One => i8::try_from(self.p1_position).expect("i8の表現範囲外") - 12,
        }
    }

    #[allow(clippy::float_arithmetic)]
    fn calc_position_reward(&self) -> f64 {
        f64::from(self.distance_from_center()) * 200.0
    }
}

impl State for MyState {
    type A = Action;

    #[allow(clippy::float_arithmetic)]
    fn reward(&self) -> f64 {
        let a = self.calc_safe_reward();
        let b = self.calc_score_reward();
        let c = self.calc_position_reward();

        // if self.round_winner == self.my_id.denote() as i8 {
        //     return 1000.0;
        // } else if self.round_winner == -1 {
        //     match self.prev_action {
        //         Some(action) => match action {
        //             Action::Attack(_) => return 50.0,
        //             Action::Move(m) => match m.direction {
        //                 Direction::Forward => return 30.0,
        //                 Direction::Back => return 10.0,
        //             }
        //         }
        //         None => return 0.0
        //     }
        // } else {
        //     return -1000.0;
        // }
        let mut won = false;
        if self.round_winner == self.my_id.denote() as i8{
            won = true;
        }
        match self.prev_action {
            Some(action) => match action {
                Action::Attack(_) => if won {
                    return 3000.0;
                } else {
                    return -500.0;
                },
                Action::Move(m) => match m.direction {
                    Direction::Forward => if won {
                        return 1500.0;
                    } else {
                        return 200.0;
                    },
                    Direction::Back => if won {
                        return 1500.0;
                    } else {
                        return -200.0
                    }
                }
            }
            None => return 0.0
        }
        
    }
    fn actions(&self) -> Vec<Action> {
        fn attack_cards(hands: &[CardID], card: CardID) -> Option<Action> {
            let have = hands.iter().filter(|&&x| x == card).count();
            (have > 0).then(|| {
                Action::Attack(Attack {
                    card,
                    quantity: Maisuu::from_usize(have).expect("Maisuuの境界内"),
                })
            })
        }
        fn decide_moves(for_back: bool, for_forward: bool, card: CardID) -> Vec<Action> {
            use Direction::{Back, Forward};
            match (for_back, for_forward) {
                (true, true) => vec![
                    Action::Move(Movement {
                        card,
                        direction: Back,
                    }),
                    Action::Move(Movement {
                        card,
                        direction: Forward,
                    }),
                ],
                (true, false) => vec![Action::Move(Movement {
                    card,
                    direction: Back,
                })],
                (false, true) => vec![Action::Move(Movement {
                    card,
                    direction: Forward,
                })],
                (false, false) => {
                    vec![]
                }
            }
        }
        if self.game_end {
            return Vec::new();
        }
        let set = self
            .hands
            .iter()
            .copied()
            .collect::<HashSet<_, RandomState>>();
        match self.my_id {
            PlayerID::Zero => {
                let moves = set
                    .into_iter()
                    .flat_map(|card| {
                        decide_moves(
                            self.p0_position.saturating_sub(card.denote()) >= 1,
                            self.p0_position + card.denote() < self.p1_position,
                            card,
                        )
                    })
                    .collect::<Vec<Action>>();
                let attack = (|| {
                    let n = self.p1_position.checked_sub(self.p0_position)?;
                    let card = CardID::from_u8(n)?;
                    attack_cards(&self.hands, card)
                })();
                [moves, attack.into_iter().collect::<Vec<_>>()].concat()
            }
            PlayerID::One => {
                let moves = set
                    .into_iter()
                    .flat_map(|card| {
                        decide_moves(
                            self.p1_position + card.denote() <= 23,
                            self.p1_position.saturating_sub(card.denote()) > self.p0_position,
                            card,
                        )
                    })
                    .collect::<Vec<Action>>();
                let attack = (|| {
                    let n = self.p1_position.checked_sub(self.p0_position)?;
                    let card = CardID::from_u8(n)?;
                    attack_cards(&self.hands, card)
                })();
                [moves, attack.into_iter().collect::<Vec<_>>()].concat()
            }
        }
    }
}
// struct MyState {
//     my_id: PlayerID,
//     hands: Vec<u8>,
//     cards: RestCards,
//     p0_score: u32,
//     p1_score: u32,
//     my_position: u8,
//     enemy_position: u8,
//     game_end: bool,
// }
impl From<MyState> for [f32; 15] {
    fn from(value: MyState) -> Self {
        let id = vec![f32::from(value.my_id.denote())];
        let hands = value
            .hands
            .into_iter()
            .map(|x| f32::from(x.denote()))
            .collect::<Vec<f32>>()
            .also(|hands| hands.resize(5, 0.0));
        let cards = value
            .used
            .get_nakami()
            .iter()
            .map(|&x| f32::from(x.denote()))
            .collect::<Vec<f32>>();
        #[allow(clippy::as_conversions)]
        #[allow(clippy::cast_precision_loss)]
        let p0_score = vec![value.p0_score as f32];
        #[allow(clippy::as_conversions)]
        #[allow(clippy::cast_precision_loss)]
        let p1_score = vec![value.p1_score as f32];
        let my_position = vec![f32::from(value.p0_position)];
        let enemy_position = vec![f32::from(value.p1_position)];
        [
            id,
            hands,
            cards,
            p0_score,
            p1_score,
            my_position,
            enemy_position,           
        ]
        .concat()
        .try_into()
        .expect("長さが15")
    }
}

/// エージェントは、先ほどの「できる行動のリスト」からランダムで選択されたアクションを実行し、状態(先ほどのState)を変更する。
#[derive(Debug)]
pub struct MyAgent {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    state: MyState,
}

impl MyAgent {
    /// エージェントを作成します。
    pub fn new(
        id: PlayerID,
        hands: Vec<CardID>,
        position_0: u8,
        position_1: u8,
        reader: BufReader<TcpStream>,
        writer: BufWriter<TcpStream>,
    ) -> Self {
        MyAgent {
            reader,
            writer,
            state: MyState {
                my_id: id,
                hands,
                used: UsedCards::new(),
                p0_score: 0,
                p1_score: 0,
                p0_position: position_0,
                p1_position: position_1,
                game_end: false,
                prev_action: None,
                round_winner: -1,
            },
        }
    }
    // pub fn has_won(&self) -> bool {
    //     // ゲームが終了していて、エージェントのスコアが相手より高ければ勝利とする
    //     self.state.game_end() && self.state.p0_score() > self.state.p1_score()
    // }
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }
    fn take_action(&mut self, &action: &Action) {
        fn send_action(writer: &mut BufWriter<TcpStream>, action: Action) -> io::Result<()> {
            match action {
                Action::Attack(a) => send_info(writer, &PlayAttack::from_info(a)),
                Action::Move(m) => send_info(writer, &PlayMovement::from_info(m)),
            }
        }
        use Messages::{
            Accept, BoardInfo, DoPlay, GameEnd, HandInfo, Played, RoundEnd, ServerError,
        };
        //selfキャプチャしたいからクロージャで書いてる
        let mut take_action_result = || -> io::Result<()> {
            loop {
                match Messages::parse(&read_stream(&mut self.reader)?) {
                    Ok(messages) => match messages {
                        BoardInfo(board_info) => {
                            (self.state.p0_position, self.state.p1_position) =
                                (board_info.p0_position(), board_info.p1_position());
                            (self.state.p0_score, self.state.p1_score) =
                                (board_info.p0_score(), board_info.p1_score());
                            self.state.round_winner = -1;
                        }
                        HandInfo(hand_info) => {
                            let hand_vec = hand_info.to_vec();
                            self.state.hands = hand_vec;
                            break;
                        }
                        Accept(_) => {}
                        DoPlay(_) => {
                            send_info(&mut self.writer, &Evaluation::new())?;
                            send_action(&mut self.writer, action)?;
                            self.state.prev_action = Some(action);
                        }
                        ServerError(e) => {
                            print("エラーもらった")?;
                            print(format!("{e:?}"))?;
                            break;
                        }
                        Played(played) => {
                            self.state.used.used_action(played.to_action());
                            break;
                        }
                        RoundEnd(round_end) => {
                            // print(
                            //     format!("ラウンド終わり! 勝者:{}", round_end.round_winner).as_str(),
                            // )?;
                            match round_end.round_winner() {
                                0 => self.state.p0_score += 1,
                                1 => self.state.p1_score += 1,
                                _ => {}
                            }
                            self.state.used = UsedCards::new();
                            self.state.round_winner = round_end.round_winner();
                            break;
                        }
                        GameEnd(game_end) => {
                            print(format!(" 勝者:{}", game_end.winner()).as_str())?;
                            print(if game_end.winner() == self.state.my_id.denote() {
                                "AIの勝ち!"
                            } else {
                                ""
                            })?;
                            // print(format!("最終報酬:{}", self.state.reward()))?;
                            // print(format!(
                            //     "safe_possibilityの寄与:{}",
                            //     self.state.calc_safe_reward()
                            // ))?;
                            self.state.game_end = true;
                            break;
                        }
                    },
                    Err(e) => {
                        panic!("JSON解析できなかった {e}");
                    }
                }
            }
            Ok(())
        };
        take_action_result().expect("正しい挙動");
    }
}
