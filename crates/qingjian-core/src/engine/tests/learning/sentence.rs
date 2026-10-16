//! 整句回灌：整段选择记独立信号、满三次回灌成用户词、撤销与删候选对称。

use super::*;

#[test]
fn committing_a_sentence_records_the_whole_scope_choice_and_reflows_into_user_words() {
    let shared = Arc::new(Mutex::new((Vec::new(), sentence::UserNgram::default())));
    let mut engine = engine().with_learner(Box::new(WordLearner {
        shared: Arc::clone(&shared),
        ..WordLearner::default()
    }));
    let commit_sentence = |engine: &mut Engine| {
        engine.set_input("xiangkaifa");
        let candidate = engine.query().unwrap().candidates.items[0].clone();
        assert_eq!(candidate.kind, CandidateKind::Sentence);
        engine.commit(&candidate);
    };
    // 前两次只记「这段拼音 → 这句话」的整段选择，不回灌；它不进按输入串的 choices
    commit_sentence(&mut engine);
    assert_eq!(
        engine.learner().choice_weight("xiangkaifa", "想开发"),
        0
    );
    assert_eq!(
        engine
            .learner()
            .sentence_choice_weight("xiangkaifa", "xiangkaifa", "想开发"),
        1
    );
    commit_sentence(&mut engine);
    engine.punctuate('，');
    assert_eq!(
        engine
            .learner()
            .sentence_choice_weight("xiangkaifa", "xiangkaifa", "想开发"),
        2
    );
    assert!(shared.lock().unwrap().0.is_empty());
    // 第三次：回灌成用户词
    commit_sentence(&mut engine);
    assert_eq!(
        engine
            .learner()
            .sentence_choice_weight("xiangkaifa", "xiangkaifa", "想开发"),
        3
    );
    assert_eq!(shared.lock().unwrap().0, ["想开发"]);
    // 下次整段打出来，用户词排第一（确定性命中），不再是概率整句
    engine.set_input("xiangkaifa");
    let first = engine.query().unwrap().candidates.items[0].clone();
    assert_eq!(first.text, "想开发");
    assert_eq!(first.kind, CandidateKind::Chinese);
    // 回灌的用户词：删整句候选能把它删掉
    let as_sentence = Candidate {
        text: "想开发".to_owned(),
        kind: CandidateKind::Sentence,
        syllables: vec!["xiang".to_owned(), "kai".to_owned(), "fa".to_owned()],
        reading: None,
        translation: None,
        aux_code: None,
    };
    assert!(engine.forget(&as_sentence).user_word);
    assert_eq!(engine.learner().weight("想开发"), 0);
}

#[test]
fn erasing_a_committed_sentence_retracts_its_whole_scope_choice() {
    let shared = Arc::new(Mutex::new((Vec::new(), sentence::UserNgram::default())));
    let mut engine = engine().with_learner(Box::new(WordLearner {
        shared: Arc::clone(&shared),
        ..WordLearner::default()
    }));
    engine.set_input("xiangkaifa");
    let sentence = engine.query().unwrap().candidates.items[0].clone();
    assert_eq!(sentence.kind, CandidateKind::Sentence);
    engine.commit(&sentence);
    assert_eq!(
        engine
            .learner()
            .sentence_choice_weight("xiangkaifa", "xiangkaifa", "想开发"),
        1
    );
    // 想开发 三个字全删掉，重打同一段拼音选了别的词：整段选择退回
    for _ in 0..3 {
        engine.note_backspace();
    }
    engine.set_input("xiangkaifa");
    let xiang = engine
        .query()
        .unwrap()
        .candidates
        .items
        .into_iter()
        .find(|c| c.text == "想" && c.kind == CandidateKind::Chinese)
        .unwrap();
    engine.commit(&xiang);
    assert_eq!(
        engine.learner().choice_weight("xiangkaifa", "想开发"),
        0
    );
}
