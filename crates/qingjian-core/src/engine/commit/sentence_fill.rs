//! 整句回灌：整段记一条「这段拼音就是要这句话」的选择，同一段拼音提交满多次的
//! 同一句话回灌成用户词；整句候选对应的词序列也从这里取。

use super::*;

impl Engine {
    /// 整句回灌：同一段拼音提交过够多次（[`AUTO_WORD_SENTENCE_THRESHOLD`]）的同一句话，存成用户词。
    /// 音节来自词图对齐（[`Self::sentence_words`] 校验过与文本逐字对应），不用信模型的读音。
    /// 下次整段打出来，词图直接命中这个整句节点、按选择次数排第一——不再靠 n-gram 概率翻盘，
    /// 这是「整句入词库」对青简原有学习链路的补强。上限 [`AUTO_WORD_SENTENCE_MAX_CHARS`] 字
    /// 与 [`MAX_WORD_SYLLABLES`] 音节：更长的进不了整句词图的格子，存了也命中不了。
    pub(super) fn finish_sentence(&mut self, input: &str, reading: &str, candidate: &Candidate) {
        let chars = candidate.text.chars().count();
        if input.is_empty()
            || chars > AUTO_WORD_SENTENCE_MAX_CHARS
            || candidate.syllables.len() > sentence::MAX_WORD_SYLLABLES
            || chars != candidate.syllables.len()
        {
            return;
        }
        if self
            .learner
            .sentence_choice_weight(input, reading, &candidate.text)
            < AUTO_WORD_SENTENCE_THRESHOLD
        {
            return;
        }
        if self.knows_word(candidate) {
            return;
        }
        tracing::debug!(text = %candidate.text, "整句多次提交，回灌成用户词");
        self.learner
            .learn_word(&candidate.text, &candidate.syllables);
    }

    /// 整句候选对应的词序列：重算一次整句转换，文本对得上才算（对不上说明候选来自别处，不记）。
    /// 末尾是英文词的整句（我想学好rust）先按头段试，英文词作为最后一个词（音节就是敲的字母）。
    pub(super) fn sentence_words(
        &self,
        candidate: &Candidate,
    ) -> Option<Vec<sentence::SentenceWord>> {
        let scope = self.composition.scope();
        if self.decode(scope).is_none()
            && let Some(tail) = self.split_english_tail(scope)
            && let Some(words) = self.mixed_words(scope, &tail, &candidate.text)
        {
            return Some(words);
        }
        let conversion = match (self.decode(scope), self.active_correction(scope)) {
            (Some(decoded), _) => {
                self.convert_sentence(&decoded.segmentation()?.patterns(), true)?
            }
            (None, Some(c)) => self.convert_sentence(&c.segmentation.patterns(), false)?,
            (None, None) => {
                let (segmentations, _) = segment_longest_prefix(scope).ok()?;
                self.convert_sentence(&segmentations.first()?.patterns(), true)?
            }
        };
        (conversion.text == candidate.text).then_some(conversion.words)
    }
}
