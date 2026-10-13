//! 各张表的路径、加载、保存与重建：用户词、按输入串的选择、个人英文词、个人敲错表、个人 n-gram。

use super::*;

impl FrequencyLearner {
    /// 个人英文词表文件与词频文件同目录。
    pub(super) fn english_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_ENGLISH_FILE)
    }

    /// 从 `词\t次数[\t天数]` 读个人英文词，返回跳过的坏行数。
    /// 旧格式没有天数列，整份按 `fallback_day`（文件的修改日）计。
    pub(super) fn load_english(&mut self, source: &str, fallback_day: u64) -> usize {
        let mut skipped = 0;
        for line in data_lines(source) {
            let mut fields = line.split('\t');
            let parsed = match (fields.next(), fields.next(), fields.next()) {
                (Some(word), Some(count), day) => {
                    let day = day.map_or(fallback_day, |day| {
                        day.trim().parse::<u64>().unwrap_or(fallback_day)
                    });
                    count
                        .trim()
                        .parse::<u32>()
                        .ok()
                        .map(|count| (word, count, day))
                }
                _ => None,
            };
            match parsed {
                Some((word, count, day)) => {
                    if count > 0 {
                        self.english.insert(
                            word.to_ascii_lowercase(),
                            (word.to_owned(), Decay::from_parts(count, day)),
                        );
                    }
                }
                None => skipped += 1,
            }
        }
        self.rebuild_english();
        skipped
    }

    /// 个人英文词很少，每次变化整个重建词表即可。有效次数当词频，补全时常用的在前；
    /// 冷却到 0 的条目不再进词表（落盘时会清出）。
    pub(super) fn rebuild_english(&mut self) {
        let kept: Vec<(&String, &(String, Decay))> = self
            .english
            .iter()
            .filter(|(_, (_, entry))| self.effective(entry) > 0)
            .collect();
        if kept.is_empty() {
            self.english_list = None;
            return;
        }
        let tsv: String = kept
            .iter()
            .map(|(code, (word, entry))| format!("{word}\t{code}\t{}\n", self.effective(entry)))
            .collect();
        match WordList::parse(&tsv) {
            Ok(list) => self.english_list = Some(list),
            Err(error) => tracing::warn!(%error, "个人英文词表重建失败"),
        }
    }

    /// 个人英文词条数。
    pub fn english_count(&self) -> usize {
        self.english.len()
    }

    pub(super) fn save_english_to(&mut self, path: &Path) -> Result<(), LearningError> {
        let now = self.today.get();
        let half_life = self.half_life_days;
        // 冷却到 0 的条目文件与内存一起清出
        self.english
            .retain(|_, (_, entry)| entry.effective(now, half_life) > 0);
        let mut rows: Vec<(&String, &(String, Decay))> = self.english.iter().collect();
        rows.sort_by(|a, b| {
            b.1.1
                .effective(now, half_life)
                .cmp(&a.1.1.effective(now, half_life))
                .then_with(|| a.1.0.cmp(&b.1.0))
        });
        write_atomic(path, |file| {
            writeln!(file, "# 青简个人英文词：词\\t次数\\t最近使用日（unix 天）")?;
            for (_, (word, entry)) in rows {
                writeln!(file, "{word}\t{}\t{}", entry.count, entry.last_used)?;
            }
            Ok(())
        })?;
        // 清出后词表同步重建，冷却到 0 的词不再出候选
        self.rebuild_english();
        self.english_dirty = false;
        Ok(())
    }

    /// 按输入串记的选择文件与词频文件同目录。
    pub(super) fn choices_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_CHOICES_FILE)
    }

    /// 从 `输入串\t词\t次数` 读按输入串记的选择，返回跳过的坏行数。
    pub(super) fn load_choices(&mut self, source: &str) -> usize {
        let mut skipped = 0;
        for line in data_lines(source) {
            let Some((input, text, count)) = parse_counted_pair(line) else {
                skipped += 1;
                continue;
            };
            if count > 0 {
                self.choices
                    .entry(input.to_owned())
                    .or_default()
                    .insert(text.to_owned(), count);
            }
        }
        skipped
    }

    pub(super) fn save_choices_to(&mut self, path: &Path) -> Result<(), LearningError> {
        let mut rows: Vec<(&String, &String, &u32)> = self
            .choices
            .iter()
            .flat_map(|(input, texts)| texts.iter().map(move |(text, count)| (input, text, count)))
            .collect();
        rows.sort_by(|a, b| {
            a.0.cmp(b.0)
                .then_with(|| b.2.cmp(a.2))
                .then_with(|| a.1.cmp(b.1))
        });
        write_atomic(path, |file| {
            writeln!(file, "# 青简按输入串记的选择：输入串\t词\t次数")?;
            for (input, text, count) in rows {
                writeln!(file, "{input}\t{text}\t{count}")?;
            }
            Ok(())
        })?;
        self.choices_dirty = false;
        Ok(())
    }

    /// 个人敲错表与词频文件同目录。
    pub(super) fn typos_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_TYPOS_FILE)
    }

    /// 整句选择文件与词频文件同目录。
    pub(super) fn sentences_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_SENTENCES_FILE)
    }

    /// 从 `输入串\t读音\t句子\t次数` 读整句选择，返回跳过的坏行数。
    pub(super) fn load_sentences(&mut self, source: &str) -> usize {
        let mut skipped = 0;
        for line in data_lines(source) {
            let mut fields = line.split('\t');
            let parsed = match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some(input), Some(reading), Some(text), Some(count)) => count
                    .trim()
                    .parse::<u32>()
                    .ok()
                    .map(|count| (input, reading, text, count)),
                _ => None,
            };
            match parsed {
                Some((input, reading, text, count)) => {
                    if count > 0 && !text.is_empty() {
                        *self
                            .sentences
                            .entry((input.to_owned(), reading.to_owned()))
                            .or_default()
                            .entry(text.to_owned())
                            .or_insert(0) += count;
                    }
                }
                None => skipped += 1,
            }
        }
        skipped
    }

    pub(super) fn save_sentences_to(&mut self, path: &Path) -> Result<(), LearningError> {
        let mut rows: Vec<(&(String, String), &String, &u32)> = self
            .sentences
            .iter()
            .flat_map(|(key, texts)| texts.iter().map(move |(text, count)| (key, text, count)))
            .collect();
        rows.sort_by(|a, b| {
            (a.0)
                .0
                .cmp(&(b.0).0)
                .then_with(|| (a.0).1.cmp(&(b.0).1))
                .then_with(|| b.2.cmp(a.2))
                .then_with(|| a.1.cmp(b.1))
        });
        write_atomic(path, |file| {
            writeln!(file, "# 青简整句选择：输入串\t读音\t句子\t次数")?;
            for (key, text, count) in rows {
                writeln!(file, "{}\t{}\t{text}\t{count}", key.0, key.1)?;
            }
            Ok(())
        })?;
        self.sentences_dirty = false;
        Ok(())
    }

    /// 整句选择条数。
    pub fn sentence_count(&self) -> usize {
        self.sentences.values().map(HashMap::len).sum()
    }

    /// 超上限时所有整句计数减半，去掉减到零的。
    pub(super) fn decay_sentences(&mut self) {
        for texts in self.sentences.values_mut() {
            texts.retain(|_, count| {
                *count /= 2;
                *count > 0
            });
        }
        self.sentences.retain(|_, texts| !texts.is_empty());
    }

    /// 从 `敲的\t要的\t次数[\t天数]` 读个人敲错表，返回跳过的坏行数。
    /// 旧格式没有天数列，整份按 `fallback_day`（文件的修改日）计。
    pub(super) fn load_typos(&mut self, source: &str, fallback_day: u64) -> usize {
        let mut skipped = 0;
        for line in data_lines(source) {
            let mut fields = line.split('\t');
            let parsed = match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some(typed), Some(intended), Some(count), day) => {
                    let day = day.map_or(fallback_day, |day| {
                        day.trim().parse::<u64>().unwrap_or(fallback_day)
                    });
                    count
                        .trim()
                        .parse::<u32>()
                        .ok()
                        .map(|count| (typed, intended, count, day))
                }
                _ => None,
            };
            match parsed {
                Some((typed, intended, count, day)) => {
                    if count > 0 {
                        self.typos
                            .entry(typed.to_owned())
                            .or_default()
                            .insert(intended.to_owned(), Decay::from_parts(count, day));
                    }
                }
                None => skipped += 1,
            }
        }
        skipped
    }

    pub(super) fn save_typos_to(&mut self, path: &Path) -> Result<(), LearningError> {
        let now = self.today.get();
        let half_life = self.half_life_days;
        // 冷却到 0 的条目文件与内存一起清出
        for intended in self.typos.values_mut() {
            intended.retain(|_, entry| entry.effective(now, half_life) > 0);
        }
        self.typos.retain(|_, intended| !intended.is_empty());
        let mut rows: Vec<(&String, &String, &Decay)> = self
            .typos
            .iter()
            .flat_map(|(typed, intended)| {
                intended
                    .iter()
                    .map(move |(syllable, entry)| (typed, syllable, entry))
            })
            .collect();
        rows.sort_by(|a, b| {
            b.2.effective(now, half_life)
                .cmp(&a.2.effective(now, half_life))
                .then_with(|| a.0.cmp(b.0))
                .then_with(|| a.1.cmp(b.1))
        });
        write_atomic(path, |file| {
            writeln!(
                file,
                "# 青简个人敲错表：敲的\t要的\t次数\t最近使用日（unix 天）"
            )?;
            for (typed, intended, entry) in rows {
                writeln!(
                    file,
                    "{typed}\t{intended}\t{}\t{}",
                    entry.count, entry.last_used
                )?;
            }
            Ok(())
        })?;
        self.typos_dirty = false;
        Ok(())
    }

    /// 个人敲错表条数。
    pub fn typo_count_total(&self) -> usize {
        self.typos.values().map(HashMap::len).sum()
    }

    /// 按输入串记的选择条数。
    pub fn choice_count(&self) -> usize {
        self.choices.values().map(HashMap::len).sum()
    }

    /// 所有按输入串记的计数减半，去掉减到零的。
    pub(super) fn decay_choices(&mut self) {
        for texts in self.choices.values_mut() {
            texts.retain(|_, count| {
                *count /= 2;
                *count > 0
            });
        }
        self.choices.retain(|_, texts| !texts.is_empty());
    }

    /// 用户词文件与词频文件同目录。
    pub(super) fn words_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_WORDS_FILE)
    }

    /// 个人 n-gram 文件与词频文件同目录。
    pub(super) fn ngram_path(frequency_path: &Path) -> PathBuf {
        frequency_path.with_file_name(USER_NGRAM_FILE)
    }

    /// 个人 n-gram 里不同的转移条数（二元对 + 三元条）。
    pub fn ngram_transition_count(&self) -> usize {
        self.ngram.transition_count()
    }

    pub(super) fn save_ngram_to(&mut self, path: &Path) -> Result<(), LearningError> {
        let tsv = self.ngram.to_tsv();
        write_atomic(path, |file| {
            writeln!(file, "# 青简个人 n-gram：前词\t后词\t次数，句首用 <s>")?;
            file.write_all(tsv.as_bytes())
        })?;
        self.ngram_dirty = false;
        Ok(())
    }

    /// 从主词库同款 TSV 读用户词，返回跳过的坏行数。
    pub(super) fn load_words(&mut self, source: &str) -> usize {
        let mut skipped = 0;
        for line in data_lines(source) {
            let mut fields = line.split('\t');
            let (Some(text), Some(pinyin)) = (fields.next(), fields.next()) else {
                skipped += 1;
                continue;
            };
            if text.is_empty() || pinyin.trim().is_empty() {
                skipped += 1;
                continue;
            }
            self.words.insert(text.to_owned(), pinyin.trim().to_owned());
        }
        self.rebuild_words();
        skipped
    }

    /// 用户词很少，每次变化整个重建小词库即可。
    pub(super) fn rebuild_words(&mut self) {
        if self.words.is_empty() {
            self.user_dictionary = None;
            return;
        }
        let tsv: String = self
            .words
            .iter()
            .map(|(text, pinyin)| format!("{text}\t{pinyin}\t{USER_WORD_FREQUENCY}\n"))
            .collect();
        match Dictionary::parse(&tsv) {
            Ok(dictionary) => self.user_dictionary = Some(dictionary),
            Err(error) => tracing::warn!(%error, "用户词库重建失败"),
        }
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub(super) fn save_words_to(&mut self, path: &Path) -> Result<(), LearningError> {
        write_atomic(path, |file| {
            writeln!(file, "# 青简用户词：词\\t拼音\\t词频，与主词库同格式")?;
            for (text, pinyin) in &self.words {
                writeln!(file, "{text}\t{pinyin}\t{USER_WORD_FREQUENCY}")?;
            }
            Ok(())
        })?;
        self.words_dirty = false;
        Ok(())
    }
}
