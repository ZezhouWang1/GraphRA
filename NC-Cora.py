import os
# os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset as Dataset2
from torch.nn import CrossEntropyLoss

from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, TrainingArguments
from transformers import AutoTokenizer , AutoConfig, AutoModelForSequenceClassification

from typing import Optional
from peft import LoraConfig
from dataclasses import dataclass, field
# from datasets import load_dataset, Dataset, DatasetDict

# from ogb.nodeproppred import PygNodePropPredDataset
# from ogb.graphproppred import GraphPropPredDataset
# from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid


# from huggingface_hub import login
# login("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")


MODEL_NAME = "huggyllama/llama-7b"  # "7B"  # "huggyllama/llama-7b"
DATASET_NAME = 'Cora'  # 'ogbn-products', 'cora', 'pubmed'
K = 5  
NUM_LABELS = 7
MAX_LENGTH = 100
BATCH_SIZE = 2
EPOCHS = 60
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
LoRA_R = 96
LoRA_ALPHA = 32


@dataclass
class ScriptArguments:
    # Setting for Model
    model_name: Optional[str] = field(default=MODEL_NAME, metadata={"help": "the model name"})

    # Setting for Training
    learning_rate: Optional[float] = field(default=LEARNING_RATE, metadata={"help": "the learning rate"})  # 1.41e-5
    batch_size: Optional[int] = field(default=BATCH_SIZE, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=GRADIENT_ACCUMULATION_STEPS, metadata={"help": "the number of gradient accumulation steps"})
    num_train_epochs: Optional[int] = field(default=EPOCHS, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})

    # Setting for LoRA
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=LoRA_R, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=LoRA_ALPHA, metadata={"help": "the alpha parameter of the LoRA adapters"})
    
    # Setting for Save
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    output_dir: Optional[str] = field(default="Output", metadata={"help": "the output directory"})
    
    # Setting for Log
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})

    # Setting for Permission
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

    # Maybe use for CausalLM
    # dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})  
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"}) 


class CustomDataset(Dataset2):
    def __init__(self, embeds, labels):
        self.embeds = embeds.squeeze(1).to(torch.bfloat16).detach()
        self.labels = labels.unsqueeze(1).detach()
        self.attention_mask = create_attention_mask(self.embeds).detach()
        
    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        item = {"inputs_embeds": self.embeds[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}
        return item


def create_attention_mask(embeds):
    last_feature_zero = embeds[:, :, -1] == 0

    # 将结果转换为整数（0或1）
    attention_mask = (~last_feature_zero).to(torch.int)
    return attention_mask


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").view(-1)
        inputs_embeds = inputs["inputs_embeds"]

        # forward pass
        outputs = model(inputs_embeds=inputs_embeds)

        logits = outputs.get("logits")
        # print(f"\n===\n logits.shape={logits.shape}; labels.shape={labels.shape}\n###\n {logits} \n ### {labels} ===\n")
        # 确保 logits 形状正确
        if logits.shape[-1] != self.model.config.num_labels:
            raise ValueError(f"Logits 的最后一个维度应为 {self.model.config.num_labels}, 但得到的是 {logits.shape[-1]}")

        # 计算损失
        loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits, labels)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss


def embed_extract(tokenizer, extract_embedding, text):
  tokens = tokenizer(text, return_tensors="pt")
  embeds = extract_embedding(tokens["input_ids"])
  return embeds


def expand_embedding(embeddings):
    # 检查是单个嵌入向量还是一批嵌入向量
    if len(embeddings.shape) == 1:
        # 处理单个 [1024] 维度的向量
        if embeddings.shape[0] != 1024:
            raise ValueError("Input embedding must be a 1D tensor with 1024 elements")
        return (embeddings / 4.0).repeat(4)
    elif len(embeddings.shape) == 2:
        # 处理 [N, 1024] 维度的批量向量
        if embeddings.shape[1] != 1024:
            raise ValueError("Each embedding in the batch must have 1024 elements")
        return (embeddings / 4.0).repeat(1, 4)
    else:
        raise ValueError("Input embedding must be either 1D or 2D")


def pad_tensor(tensor, pad_size, dim, pad_value=0):
    """将张量填充到指定的大小。"""
    pad = (0, 0) * (tensor.dim() - dim - 1) + (0, pad_size - tensor.size(dim))
    return torch.nn.functional.pad(tensor, pad, 'constant', 0)


def construct_instruction(node_idx, node_feat, K, K_idx, K_feat, node_target, tokenizer, extract_embedding):
    query_part_1 = f"Central node [{node_idx}] is featured with text feature"
    query_part_2 = f"are the top-{K} important neighbors nodes {K_idx}'s features within two-hops."

    embed_1 = embed_extract(tokenizer, extract_embedding, query_part_1)
    embed_2 = embed_extract(tokenizer, extract_embedding, query_part_2)
    # embed_3 = embed_extract(tokenizer, extract_embedding, query_part_3)

    node_feat = expand_embedding(node_feat).view(1, 1, -1)
    K_feat = expand_embedding(K_feat).view(1, K, -1)

    device = embed_1.device
    node_feat = node_feat.to(device)
    K_feat = K_feat.to(device)

    instrcution_embedding = torch.cat((embed_1, node_feat, K_feat, embed_2), dim=1)
    # instrcution_embedding = torch.cat((embed_1, node_feat, K_feat, embed_2, embed_3), dim=1)

    if instrcution_embedding.size(1) < MAX_LENGTH:
        instrcution_embedding = pad_tensor(instrcution_embedding, MAX_LENGTH, dim=1, pad_value=0)

    answer = int(node_target)

    return instrcution_embedding.cpu(), answer


def load_model():
    script_args = ScriptArguments()
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        device_map = "auto"  #{"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    # 加载预训练模型的配置
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_LABELS

    # 加载预训练模型，并添加分类头
    model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    config=config,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer, script_args


def load_data(x_embs_file, top_k_neighbors_file, dataset_type='train'):
    # 加载嵌入和邻居索引
    x_embs = torch.load(x_embs_file)
    # 使用 JSON 加载 top_k_neighbors 数据
    with open(top_k_neighbors_file, 'r') as file:
        top_k_neighbors = json.load(file)

    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name=DATASET_NAME)
    data = dataset[0]

    if dataset_type == 'train':
        set_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif dataset_type == 'valid':
        set_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    elif dataset_type == 'test':
        set_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    else:
        raise ValueError(f"Invalid set name: {dataset_type}. Must be one of ['train', 'valid', 'test']")

    data_list = []

    for position, idx in tqdm(enumerate(set_idx), desc=f"Processing {dataset_type} Data"):
        node_idx = idx.item()  # 节点索引
        node_feat = x_embs[node_idx]  # 节点特征
        # 使用位置索引获取top_k_neighbors
        k_idx = top_k_neighbors[str(position)]   #top_k_neighbors[str(node_idx)]  # 使用位置索引
        k_feat = x_embs[k_idx]  # 邻居特征
        node_target = data.y[idx]  # 节点目标/标签

        data_list.append({
            'node_idx': node_idx,
            'node_feat': node_feat,
            'K_idx': k_idx,
            'K_feat': k_feat,
            'label': node_target
        })


    return data_list


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("\n======")
    print(labels.squeeze())
    print(predictions)
    print("======\n")
    return {"accuracy": accuracy_score(labels, predictions)}


def main():
    llama2_model, llama2_tokenizer, script_args = load_model()
    extract_embedding = llama2_model.get_input_embeddings()

    # 使用函数生成数据
    embs_path = 'Data/Cora/simteg_roberta_x.pt'
    train_top_k_neighbors_path = 'Data/Cora/train_top_{K}_neighbors.json'
    # valid_top_k_neighbors_path = 'Data/valid_top_k_neighbors.json'
    test_top_k_neighbors_path='Data/Cora/test_top_{K}_neighbors.json'

    # try:
    # 直接加载 .pt 文件
        # train_data=torch.load("Instruction-Cora/train_data.pt")
        # valid_data=torch.load("Instruction/valid_data.pt")
        # test_data=torch.load("Instruction-Cora/test_data.pt")
    # except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
    train_data = load_data(embs_path, train_top_k_neighbors_path, 'train')
        # valid_data = load_data(embs_path, valid_top_k_neighbors_path, 'valid')
    test_data = load_data(embs_path, test_top_k_neighbors_path, 'test')
        # torch.save(train_data, "Instruction-Cora/train_data.pt")
        # torch.save(valid_data, "Instruction/valid_data.pt")
        # torch.save(test_data, "Instruction-Cora/test_data.pt")


    # try:
    # 直接加载 .pt 文件
       #  train_instructions = torch.load("Instruction-Cora/train_instructions.pt")
    # except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
    train_instructions = [construct_instruction(node['node_idx'], node['node_feat'], K, node['K_idx'], node['K_feat'], node['label'], llama2_tokenizer, extract_embedding)
            for node in tqdm(train_data, desc='Processing Train Instructions')]
        # torch.save(train_instructions, "Instruction-Cora/train_instructions.pt")

    # try:
    # # 直接加载 .pt 文件
    #     valid_instructions = torch.load("Instruction/valid_instructions.pt")
    # except FileNotFoundError:
    # # 如果文件不存在，则处理数据并保存
    #     valid_instructions = [construct_instruction(node['node_idx'], node['node_feat'], K, node['K_idx'], node['K_feat'], node['label'], llama2_tokenizer, extract_embedding)
    #               for node in tqdm(valid_data, desc='Processing Validation Instructions')]
    #     torch.save(valid_instructions, "Instruction/valid_instructions.pt")


    # try:
    # 直接加载 .pt 文件
        # test_instructions = torch.load("Instruction-Cora/test_instructions.pt")
    # except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
    test_instructions = [construct_instruction(node['node_idx'], node['node_feat'], K, node['K_idx'], node['K_feat'], node['label'], llama2_tokenizer, extract_embedding)
            for node in tqdm(test_data, desc='Processing Validation Instructions')]
        # torch.save(test_instructions, "Instruction-Cora/test_instructions.pt")

    random.shuffle(train_instructions)
    random.shuffle(test_instructions)

    train_embeds, train_labels = zip(*train_instructions)
    test_embeds, test_labels = zip(*test_instructions)


    # 转换为 torch.tensor 并确保数据在 CPU 上
    train_embeds = torch.stack(train_embeds).detach()
    train_labels = torch.tensor(train_labels, dtype=torch.long).detach()
    test_embeds = torch.stack(test_embeds).detach()
    test_labels = torch.tensor(test_labels, dtype=torch.long).detach()

    # 创建 MyDataset 实例
    train_dataset = CustomDataset(train_embeds, train_labels)
    test_dataset = CustomDataset(test_embeds, test_labels)


    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        evaluation_strategy = "epoch",
        dataloader_num_workers=50,
        # save_strategy = "epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
    )

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            # target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head'],  # Select LoRA tuning modules.
            bias="none",
            task_type= "SEQ_CLS",  #"CAUSAL_LM", FEATURE_EXTRACTION, QUESTION_ANS, SEQ_2_SEQ_LM, SEQ_CLS, TOKEN_CLS"
        )
    else:
        peft_config = None

    llama2_model.add_adapter(peft_config)

    trainer = CustomTrainer(
    # trainer = Trainer(
        model=llama2_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # trainer.save_model(training_args.output_dir)

    metrics=trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
