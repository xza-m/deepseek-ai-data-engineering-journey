"""集中维护实验数据契约和特殊 Token。"""

SCHEMA_VERSION = "0.1"
PIPELINE_VERSION = "0.1.0"
RUN_SCHEMA_VERSION = "0.1"
TRAINING_VERSION = "0.1.0"

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Byte-level BPE 需要覆盖 256 个字节，再加上项目定义的特殊 Token。
MIN_BYTE_BPE_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)
