import torch
import torch.optim as optim
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import json
from pathlib import Path
import os
import datetime
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

os.system('pip install tiktoken fairscale fire blobfile')

from data.dataset_llm import StellarDatasetManager
from nn.state_space_llm import PhysicsInformedEvolutionaryLlama, apply_lora_to_model
from nn.multimodal import setup
from llama3.llama.model_single_gpu import Transformer, ModelArgs
from llama3.llama.tokenizer import Tokenizer
from src.run_llm import *
from nn.train import *
from util.utils import *


def main():
    checkpoint_path = '/data/TalkingLatents/checkpoints/state_space_llm_alpha_1.0_beta_1.0_gamma_1.0_freeze_lora.pth'
    with open("/data/TalkingLatents/src/llm_config.json", "r") as f:
        config = json.load(f)
    local_rank, world_size, gpus_per_node = setup()
    device = torch.cuda.current_device()
    
    print(f"Distributed training setup complete. Local rank: {local_rank}, World size: {world_size}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    model, data_manager, optimizer, scheduler, tokenizer = initialize_llm(device, config)
    if config['load_checkpoints']:
        model = load_checkpoints_ddp(model, config['checkpoint_path'])
    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    b_size = config['batch_size']
    train_loader = data_manager.get_dataloaders(batch_size=b_size,
     num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), world_size=world_size)['train']
    val_loader = data_manager.get_dataloaders(batch_size=b_size,
     num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), world_size=world_size)['val']
    test_loader = data_manager.get_dataloaders(batch_size=b_size,
     num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), world_size=world_size)['test']

    trainer = LLMTrainer(model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         criterion=None,
                         train_dataloader=train_loader,
                         val_dataloader=val_loader,
                         device=device,
                         alpha=config['alpha'],
                         beta=config['beta'],
                         gamma=config['gamma'],
                         lora_params=config['lora_params'],
                         max_iter=np.inf,
                         log_path='/data/TalkingLatents/checkpoints',
                         exp_name=f"state_space_llm_alpha_{config['alpha']}_beta_{config['beta']}_gamma_{config['gamma']}_freeze_{config['lora_params']['freeze_strategy']}")

    if config['train']:
        print("Starting training...")
        trainer.fit(num_epochs=config['num_epochs'], device=device, early_stopping=10)
    trainer.predict(test_loader, device, tokenizer, max_new_tokens=375)


def test_token_length():
    with open('/data/TalkingLatents/data/dataset/stellar_evolution_results_complete.json', 'r') as f:
        results = json.load(f)
    tokenizer = Tokenizer(
        model_path=TOKENIZER_PATH,
        custom_special_tokens=custom_special_tokens
    )
    answer_lengths = []
    for (i, result) in enumerate(results['results']):
        print(i, result.keys())
        print(result['information'].keys())
        if 'Answer' in result['result']:
            # Tokenize the answer to get length
            tokens = tokenizer.encode(result['result']['Answer'], bos=True, eos=False, allowed_special="all")
            answer_lengths.append(len(tokens))
            if i > 1000:
                break

    print(f"Answer length stats:")
    print(f"Mean: {np.mean(answer_lengths):.1f} tokens")
    print(f"95th percentile: {np.percentile(answer_lengths, 95):.1f} tokens")
    print(f"Max: {max(answer_lengths)} tokens")

if __name__ == "__main__":
    # test_token_length()
    # exit()
    main()

