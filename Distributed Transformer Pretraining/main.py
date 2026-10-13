from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast

import argparse
import os
import wandb
from torch import bfloat16
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


class FeedForward(nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.ff_layernorm = nn.LayerNorm(dmodel)
        self.w1 = nn.Linear(dmodel, 4 * dmodel, bias=True)
        self.w2 = nn.Linear(dmodel, 4 * dmodel, bias=True)
        self.w3 = nn.Linear(4 * dmodel, dmodel, bias=True)

    def forward(self, x):
        x = self.ff_layernorm(x)
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x2) * x1
        x = self.w3(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
    batch_size,
    sequence_length,
    rank,
    world_size,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
    distributed=False,
):
    if split == "train":
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/train")
    else:
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/validation")
    
    
    hf_dataset = hf_dataset.shard(
        num_shards=world_size,
        index=rank,
    )
    hf_dataset = hf_dataset.to_iterable_dataset()
    if split == "train":
        hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed + rank)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
    model.eval()
    valid_losses = []
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
    mean_valid_loss = torch.tensor(
        sum(valid_losses) / len(valid_losses),
        device=device,
    )
    print(f"Local mean valid loss: {mean_valid_loss.item()}, Rank: {dist.get_rank()}")
    dist.all_reduce(mean_valid_loss, op=dist.ReduceOp.SUM)
    global_mean = mean_valid_loss.item() / dist.get_world_size()
    model.train()
    return global_mean

def wsd_lr_lambda(step, total_steps):
    warmup_steps = int(0.01 * total_steps)
    decay_steps = int(0.1 * total_steps)
    stable_steps = total_steps - warmup_steps - decay_steps

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return 1.0
    else:  
        decay_step = step - warmup_steps - stable_steps
        return max(0.0, 1.0 - decay_step / max(1, decay_steps))

def train_model(config, device, rank, world_size):

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    dataloader = get_dataloader(config.batch_size, config.seq_length, rank, world_size, split="train", distributed=True)
    train_sampler = dataloader.sampler
    valid_dataloader = get_dataloader(config.batch_size, config.seq_length, rank, world_size, split="validation", distributed=True)
    
    validation_steps = int(1e06 // (config.batch_size * config.seq_length)) # we want to evaluate on 1M tokens
    model = Transformer(config).to(device)
    model = FSDP(model, mixed_precision=mp)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: wsd_lr_lambda(step, config.train_steps)
    )

    model.train()

    for i, batch in zip(range(config.train_steps), dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"]

        optimizer.zero_grad()
        outputs = model(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()
        
        if rank == 0:
            wandb.log({"loss": loss})
        
        if i % config.log_train_loss_freq == 0:
            print(f"Step:{i}, Train Loss:{loss}")
            

        if i % config.log_valid_loss_freq == 0:
            valid_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
            if rank == 0:
                wandb.log({"valid_loss": valid_loss})
                print(f"Valid loss:{valid_loss}")

        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if rank == 0:
            wandb.log({"lr": lr})

    valid_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
    if rank == 0:
        wandb.log({"valid_loss": valid_loss})
        print(f"Final valid loss:{valid_loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    
    config = SimpleNamespace(
        train_steps=args.train_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        dropout=args.dropout,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        log_train_loss_freq=100,
        log_valid_loss_freq=100
    )
    dist.init_process_group("nccl")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("WARNING: GPU not available, running on CPU.")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Initialized process group with world size {world_size}, rank {rank}, device {device}")
    if rank == 0:
        wandb.init(
            entity="g-kicinski-university-of-warsaw",
            project="bml-project2",
            config=vars(config),
        )

    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.") 
    train_model(config, device, rank, world_size)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
