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
use rurel::{mdp::{Agent, State}, strategy::learn::q};

use crate::{
    algorithm::{safe_possibility, ProbabilityTable},
    print,
    protocol::{Evaluation, Messages, PlayAttack, PlayMovement, PlayerID},
    read_stream, send_info, Action, Attack, CardID, Direction, Maisuu, Movement, RestCards,
    HANDS_DEFAULT_U8,
};

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum GameResult {
    Win,
    Lose,
    Even,
}


/// Stateは、結果状態だけからその評価と次できる行動のリストを与える。
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct MyState {
    my_id: PlayerID,
    hands: Vec<CardID>,
    cards: RestCards,
    p0_score: u32,
    p1_score: u32,
    p0_position: u8,
    p1_position: u8,
    game_end: bool,
    prev_action: Option<Action>,
    current_result: GameResult,
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

    /// `RestCards`を返します。
    pub fn rest_cards(&self) -> RestCards {
        self.cards
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
        cards: RestCards,
        p0_score: u32,
        p1_score: u32,
        p0_position: u8,
        p1_position: u8,
        game_end: bool,
        prev_action: Option<Action>,
        current_result: GameResult,
    ) -> Self {
        Self {
            my_id,
            hands,
            cards,
            p0_score,
            p1_score,
            p0_position,
            p1_position,
            game_end,
            prev_action,
            current_result,
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

    pub fn distance_opposite(&self) -> u8 {
        self.p1_position - self.p0_position
    }
//ネスに聞く
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
            .mul(100.0)
    }

    #[allow(clippy::float_arithmetic)]
    fn calc_score_reward(&self) -> f64 {
        (f64::from(self.my_score()) * 10.0).powi(2) - (f64::from(self.enemy_score()) * 10.0).powi(2)
    }

    fn distance_from_center(&self) -> i8 {
        match self.my_id {
            PlayerID::Zero => i8::try_from(self.p0_position).expect("i8の表現範囲外") - 12,
            PlayerID::One => 12 - i8::try_from(self.p1_position).expect("i8の表現範囲外"),
        }
    }

    fn distance_between_enemy(&self) -> u8 {
        self.p1_position - self.p0_position
    }

    #[allow(clippy::float_arithmetic)]
    fn calc_position_reward(&self) -> f64 {
        f64::from(self.distance_from_center()) * 200.0
    }

    fn calc_winner_reward(&self) -> f64 {
        match self.round_winner {
            None | Some(None) => 0.0,
            Some(Some(n)) => {
                if n == self.my_id {
                    10000.0
                } else {
                    -10000.0
                }
            }
        }
    }

    fn action_reward(&self) -> f64 {
        match self.prev_action {
            None => 0.0,
            Some(action) => match action {
                Action::Move(m) => match m.direction() {
                    Direction::Forward => match self.round_winner {
                        None | Some(None) => 500.0,
                        Some(Some(player)) if player == self.my_id() => 2000.0,
                        Some(Some(_)) => -1500.0,
                    },
                    Direction::Back => match self.round_winner {
                        None | Some(None) => 0.0,
                        Some(Some(player)) if player == self.my_id() => 1000.0,
                        Some(Some(_)) => -5000.0,
                    },
                },
                Action::Attack(_) => match self.round_winner {
                    None | Some(None) => 700.0,
                    Some(Some(player)) if player == self.my_id() => 3000.0,
                    Some(Some(_)) => -1000.0,
                },
            },
        }
    }

    fn to_evaluation(&self) -> Evaluation {
        let actions = self
            .actions()
            .into_iter()
            .filter(|action| !matches!(action, Action::Attack(_)))
            .collect::<Vec<Action>>();
        let card_map = card_map_from_hands(self.hands()).expect("安心して");
        let distance = self.distance_opposite();
        let rest_cards = self.used_cards().to_restcards(card_map);
        let hands = self.hands();
        let table = &ProbabilityTable::new(&self.used_cards().to_restcards(card_map));
        let safe_sum = actions
            .iter()
            .map(|&action| {
                safe_possibility(distance, rest_cards, hands, table, action)
                    .unwrap_or(Ratio::<u64>::zero())
            })
            .sum::<Ratio<u64>>();
        let mut evaluation_set = Evaluation::new();
        if safe_sum == Ratio::zero() {
            return evaluation_set;
        }
        actions
            .into_iter()
            .map(|action| {
                (
                    action,
                    safe_possibility(distance, rest_cards, hands, table, action)
                        .unwrap_or(Ratio::zero())
                        / safe_sum,
                )
            })
            .for_each(|(action, eval)| evaluation_set.update(action, eval));
        evaluation_set
    }
}

impl State for MyState {
    type A = Action;
    #[allow(clippy::float_arithmetic)]
    fn reward(&self) -> f64 {
        let a = self.calc_safe_reward();
        let b = self.calc_score_reward();
        // let b = 0.0;
        let c = self.calc_position_reward();
        let d = self.calc_winner_reward();
        let e = self.action_reward();
        // a + c + d
        b
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
impl From<MyState> for [f32; 13] {
    #[allow(clippy::float_arithmetic)]
    fn from(value: MyState) -> Self {
        // プレイヤーIDをf32値に変更
        let id = vec![f32::from(value.my_id.denote())];
        // 自分の手札(Vec)をf32値に変更、そのままVecとして表現
        let hands = value
            .hands
            .into_iter()
            .map(|x| f32::from(x.denote() - 1) / 4.0)
            .collect::<Vec<f32>>()
            .also(|hands| hands.resize(5, 0.0));
        // 使われたカード(インデックスとカード番号が対応、値と枚数が対応)をf32値に変更、そのままVecとして表現
        let cards = value
            .used
            .into_iter()
            .iter()
            .map(|&x| f32::from(x.denote()) / 5.0)
            .collect::<Vec<f32>>();
        // プレイヤー0の位置をf32値に変更
        let my_position = vec![f32::from(value.p0_position - 1) / 22.0];
        // プレイヤー1の位置をf32値に変更
        let enemy_position = vec![f32::from(value.p1_position - 1) / 22.0];
        // 単一の配列としてまとめる
        [id, hands, cards, my_position, enemy_position]
            .concat()
            .try_into()
            .expect("長さが13")
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
                cards: RestCards::new(),
                p0_score: 0,
                p1_score: 0,
                p0_position: position_0,
                p1_position: position_1,
                game_end: false,
                prev_action: None,
                current_result: GameResult::Even,
            },
        }
    }
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }
    fn take_action(&mut self, &action: &Action) {
        fn send_action(writer: &mut BufWriter<TcpStream>, action: Action) -> io::Result<()> {
            match action {
                Action::Move(m) => send_info(writer, &PlayMovement::from_info(m)),
                Action::Attack(a) => send_info(writer, &PlayAttack::from_info(a)),
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
                            self.state.current_result = GameResult::Even;
                        }
                        HandInfo(hand_info) => {
                            let hand_vec = hand_info.to_vec().also(|hand_vec| hand_vec.sort());
                            self.state.hands = hand_vec;
                            break;
                        }
                        Accept(_) => {}
                        DoPlay(_) => {
                            send_info(&mut self.writer, &self.state.to_evaluation())?;
                            send_action(&mut self.writer, action)?;
                            self.state.used.used_action(action);
                            self.state.prev_action = Some(action);
                        }
                        ServerError(e) => {
                            print("エラーもらった")?;
                            print(format!("{e:?}"))?;
                            break;
                        }
                        Played(played) => {
                            self.state.used.used_action(played.to_action());
                        }
                        RoundEnd(round_end) => {
                            // print(
                            //     format!("ラウンド終わり! 勝者:{}", round_end.round_winner()).as_str(),
                            // )?;
                            match round_end.round_winner() {
                                0 => self.state.p0_score += 1,
                                1 => self.state.p1_score += 1,
                                _ => {}
                            }
                            self.state.used = UsedCards::new();
                            self.state.current_result = if round_end.round_winner() == self.state.my_id.denote() as i8 {
                                GameResult::Win
                            } else if round_end.round_winner() == -1 {
                                GameResult::Even
                            } else {
                                GameResult::Lose
                            };
                            break;
                        }
                        GameEnd(game_end) => {
                            self.state.round_winner = Some(PlayerID::from_u8(game_end.winner()));
                            self.state.game_end = true;
                            print(format!("ゲーム終わり! 勝者:{}", game_end.winner()).as_str())?;
                            print(if game_end.winner() == self.state.my_id.denote() {
                                "勝ちました!"
                            } else {
                                "負けました!"
                            })?;
                            print(format!("最終報酬:{}", self.state.reward()))?;
                            print(format!("p0の位置:{}", self.state.p0_position))?;
                            print(format!("p1の位置:{}", self.state.p1_position))?;
                            print(format!(
                                "position_reward:{}",
                                self.state.calc_position_reward()
                            ))?;
                            print(format!(
                                "safe_possibilityの寄与:{}",
                                self.state.calc_safe_reward()
                            ))?;
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
