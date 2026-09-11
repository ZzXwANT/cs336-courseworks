import os
import regex as re
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool

# GPT-2匹配
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 单字节缓存表 
BYTE_LOOKUP = [bytes([i]) for i in range(256)]

def pre_tokenization(input_path: str | os.PathLike,
                    num_processes: int = 4,
                    special_tokens: list[str] = ["<|endoftext|>"]
) -> Counter:
    
    if special_tokens is None:
        special_tokens = []
    
    # 只处理第一个ST
    sToken = special_tokens[0]
    pattern = re.escape(sToken) # 需要使用 re.escape 来转义特殊字符，如 <, >, | 等
    # 处理所有ST
    # pattern = "|".join(re.escape(st) for st in special_tokens)
    
    # 多线程处理文本
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, sToken.encode("utf-8"))
        
    tasks = [
        (input_path, start, end, pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(_pre_tokenization_worker, tasks)
        
    total_counts = Counter()
    for chunk_counts in results:
        total_counts.update(chunk_counts)
        
    return total_counts

def _pre_tokenization_worker(input_path, start, end, pattern):
    """子进程 worker：独立打开文件，只读取 [start, end) 范围并统计词频"""
    # 字节级文本容器
    counts = Counter()
    
    with open(input_path, "rb") as f:
        f.seek(start)
        # 解码成字符
        chunk_str = f.read(end - start).decode("utf-8", errors="ignore")
        
    # 以special_tokens分割文本
    segments = re.split(pattern, chunk_str)
    
    for seg in segments:
        for m in re.finditer(PAT, seg):
            token_bytes = m.group().encode("utf-8")
            # 查表统计(counter类方便统计)
            counts[tuple(BYTE_LOOKUP[b] for b in token_bytes)] +=1
                
    return counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    把文件切成若干块，使每块可以被独立统计。
    如果边界发生重叠，实际返回的块数可能比期望的少。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件的总字节数
    file.seek(0, os.SEEK_END) # 指针到文件末尾
    file_size = file.tell()
    file.seek(0)              # 回到开头

    # 每块的粗略大小（整除，最后一小块会补齐到文件末尾）
    chunk_size = file_size // desired_num_chunks

    # chunks边界位置划分
    # 均匀地初步猜测边界位置：共 desired_num_chunks+1 个点，首尾分别是 0 和 file_size
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096   # 每次向前读取 4KB 的小块

    # 逐个微调内部边界（跳过首尾两个边界）
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)    # 从猜测的边界位置开始读
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 读到文件末尾（读不到数据了），说明该边界应该放在文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在这小块数据中查找 special token（如 <|endoftext|>）
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到了，把边界精确挪到 token 的起始位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            # 没找到，向后移动一个 mini_chunk_size，继续读下一块
            initial_position += mini_chunk_size

    # 去重并排序，保证所有边界唯一；实际块数可能少于 desired_num_chunks
    return sorted(set(chunk_boundaries))

# *test
if __name__ == "__main__":
    from pathlib import Path
    
    # 指向 fixtures 中的 corpus.en
    corpus_path = Path(__file__).resolve().parent.parent.parent.parent /"original-llm" /"assignment1-basics" /"tests" / "fixtures" / "corpus.en"
    
    print(f"Testing pre_tokenization with: {corpus_path}")
    counts = pre_tokenization(
        input_path=corpus_path,
        num_processes=4,
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"Total unique words/chunks: {len(counts)}")
    print("Top 10 most common token sequences:")
    for seq, count in counts.most_common(10):
        # 将 tuple of bytes 转换成可见字符方便查看
        readable = [b.decode('utf-8', errors='replace') for b in seq]
        print(f"  {readable} -> {count}")
