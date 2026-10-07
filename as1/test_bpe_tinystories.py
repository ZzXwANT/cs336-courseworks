from as1.train_bpe import train_bpe
from pathlib import Path
import json
import time
import tracemalloc
from as1.common import gpt2_bytes_to_unicode

DATA_PATH = Path("/Users/mac/proj/CS336/original-llm/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    tracemalloc.start()
    start_time = time.time()
    
    vocab, merges = train_bpe(
        input_path=DATA_PATH,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=8,
    )
    end_time = time.time()
    print(f"Training BPE took {(end_time - start_time) / 60:.2f} minutes.")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()

    # 获取 256 字节到 unicode 字符串的映射字典
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_save = {idx: "".join(byte_encoder[b] for b in token) for idx, token in vocab.items()}
    with open(OUTPUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_save, f, ensure_ascii=False, indent=4)
        
    # 修复了 merge 的保存方式，确保每个 token 都是可读的 unicode 字符串
    merges_save = [
        f"{''.join(byte_encoder[b] for b in token1)} {''.join(byte_encoder[b] for b in token2)}"
        for token1, token2 in merges
    ]
    with open(OUTPUT_DIR / "merges.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(merges_save))
    
    longest_token_idx, longest_token = max(vocab.items(), key=lambda item: len(item[1]))
    
    print(f"Longest token: {longest_token.decode('utf-8', errors='ignore')} (Index: {longest_token_idx}, Length: {len(longest_token)})")