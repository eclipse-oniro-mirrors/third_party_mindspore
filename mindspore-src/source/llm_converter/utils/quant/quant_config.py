from dataclasses import dataclass


class QuantizationConfig:
    def __init__(self, quant_method: str):
        if quant_method is None:
            self.is_quant = False
            self.bits = 0
            self.group_size = 0
            return

        if quant_method not in ["W4A8", "W2A16", "W4A16"]:
            raise ValueError(
                f"`quant_method` should be one of [W4A8, W2A16, W4A16], but is {quant_method}"
            )

        self.is_quant = True
        self.quant_method = quant_method
        if quant_method == "W4A8":
            self.bits = 4
            self.group_size = 128
        elif quant_method == "W2A16":
            self.bits = 2
            self.group_size = 32
        elif quant_method == "W4A16":
            self.bits = 4
            self.group_size = 32

    def asdict(self):
        if self.is_quant:
            return {
                "quant_method": self.quant_method,
                "bits": self.bits,
                "group_size": self.group_size,
            }

    def __repr__(self):
        config = f"<is_quant: {self.is_quant}"
        if hasattr(self, "quant_method"):
            config += f", quant_method: {self.quant_method}, bits: {self.bits}, group_size: {self.group_size}"
        config += ">"
        return config


@dataclass
class ModelConfig:
    max_length: int
    chunk_size: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    eos_id: int
    embedding_quant: QuantizationConfig
    decoder_quant: QuantizationConfig

    def asdict(self):
        fields = {}
        for item in self.__dataclass_fields__.values():
            if item.type == QuantizationConfig:
                fields[item.name] = getattr(self, item.name).asdict()
            else:
                fields[item.name] = getattr(self, item.name)
        return fields


@dataclass
class LiteTurboConfig:
    max_length: int
    chunk_size: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    eos_id: int
    scale_gp_size: int
    embedding_quant: bool
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    diversity_penalty: float
    repetition_penalty: float
    length_penalty: float
    random_seed: int

    def asdict(self):
        fields = {}
        for item in self.__dataclass_fields__.values():
            fields[item.name] = getattr(self, item.name)
        return fields
