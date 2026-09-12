import os
from .pretokenization import pre_tokenization
from collections import Counter
from collections import defaultdict

def merge_word(word: list[bytes], best_pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
    """
    在给定的word中，将best_pair替换成merged。
    """
    i = 0
    new_word = []
    while i < len(word):
        if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word

def count_pairs(pre_tokens: Counter) -> Counter:
    """
    计算所有单词中相邻符号对的出现次数，并按单词频率加权。
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    
    for idx, (word, freq) in enumerate(pre_tokens.items()):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)
            
    return pair_counts, pair_to_words

def merge_pair_default(pre_tokens: Counter, best_pair: tuple[bytes, bytes], merged: bytes) -> Counter:
    """
    在每个 word 里把 best_pair（非重叠、从左到右）替换成 merged，返回新词表。
    """
    new_pre_tokens = Counter()
    for word, freq in pre_tokens.items():
        new_word = merge_word(list(word), best_pair, merged)
        new_pre_tokens[tuple(new_word)] += freq
    return new_pre_tokens

def merge_pair_advanced(words: list[list[bytes]], word_freqs: list[int], pair_counts: Counter, pair_to_words: dict[tuple[bytes, bytes], set[int]], best_pair: tuple[bytes, bytes], merged: bytes) -> None:
    """
    在每个 word 里把 best_pair（非重叠、从左到右）替换成 merged，返回新词表。
    只处理包含 best_pair 的单词，避免不必要的遍历。
    """
    # 筛选出包含 best_pair 的单词索引
    for idx in list(pair_to_words[best_pair]):
        word = words[idx]
        freq = word_freqs[idx]
        n = len(word)
        
        # 扣除原词的旧 pair 计数
        for i in range(n - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] -= freq
            
        # 替换词
        new_word = merge_word(word, best_pair, merged)
        words[idx] = new_word

        # 更新新词的 pair 计数
        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i + 1])
            pair_to_words[p].add(idx)
            pair_counts[p] += freq
            
    return  words, pair_counts, pair_to_words

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 获取pre-token（词级），Counter: tuple[bytes, ...]
    num_processes = kwargs.get("num_processes", 4)
    pre_tokens = pre_tokenization(input_path, num_processes, special_tokens)
    # ("t", "h", "e") 1000
    # 构建初始vocab：先放special tokens，再放256个单字节
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    merges: list[tuple[bytes, bytes]] = []

    # 需要合并次数 = 目标词表大小 - 256个单字节 - special tokens数量
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # 转换成 list 方便根据 idx 原地修改
    words = [list(w) for w in pre_tokens.keys()]
    word_freqs = list(pre_tokens.values())
    # 统计所有相邻pair的出现次数
    pair_counts, pair_to_words = count_pairs(pre_tokens)
    
    for _ in range(num_merges):
        if not pair_counts:
            break

        # 找到出现频率最高的pair；频率并列时取lexicographically更大的pair
        best_pair = max(
            pair_counts, 
            key=lambda p: (pair_counts[p], p)
        )
        # 更新merges
        merges.append(best_pair)
        
        # 更新vocab
        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged

        # 删除best_pair
        del pair_counts[best_pair]

        # 更新词表
        words, pair_counts, pair_to_words = merge_pair_advanced(words, word_freqs, pair_counts, pair_to_words, best_pair, merged)
        
    return vocab, merges
