import os
from .pretokenization import pre_tokenization
from collections import Counter


def count_pairs(pre_tokens: Counter) -> Counter:
    """
    计算所有单词中相邻符号对的出现次数，并按单词频率加权。
    """
    pair_counts = Counter()
    for word, freq in pre_tokens.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts

def merge_pair(pre_tokens: Counter, best_pair: tuple[bytes, bytes], merged: bytes) -> Counter:
    """
    在每个 word 里把 best_pair（非重叠、从左到右）替换成 merged，返回新词表。
    """
    new_pre_tokens = Counter()
    for word, freq in pre_tokens.items():
        n = len(word)
        new_word = []
        i = 0
        while i < n:
            if i + 1 < n and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_pre_tokens[tuple(new_word)] += freq
    return new_pre_tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 获取pre-token（词级），Counter: tuple[bytes, ...]
    num_processes = kwargs.get("num_processes", 4)
    pre_tokens = pre_tokenization(input_path, num_processes, special_tokens)

    # 构建初始vocab：先放special tokens，再放256个单字节
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    merges: list[tuple[bytes, bytes]] = []

    # 需要合并次数 = 目标词表大小 - 256个单字节 - special tokens数量
    num_merges = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges):
        # 统计所有相邻pair的出现次数
        pair_counts = count_pairs(pre_tokens)
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

        # 更新词表
        pre_tokens = merge_pair(pre_tokens, best_pair, merged)
        

    return vocab, merges
