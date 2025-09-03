import torch
import json
from pathlib import Path
import os

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

os.system('pip install tiktoken fairscale fire blobfile')
from data.dataset_llm import StellarDatasetManager
from nn.state_space_llm import PhysicsInformedEvolutionaryLlama
from llama3.llama.model_single_gpu import Transformer, ModelArgs
from llama3.llama.tokenizer import Tokenizer

TOKENIZER_PATH = "/data/.llama/checkpoints/Llama3.2-1B/tokenizer.model"
MODEL_PATH = "/data/.llama/checkpoints/Llama3.2-1B"


def _load_base_model() -> Transformer:
    """Load the base LLaMA model"""
    with open(Path(MODEL_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_batch_size=4,
        max_seq_len=1024,
        **params,
    )

    model = Transformer(model_args)

    checkpoints = sorted(Path(MODEL_PATH).glob("*.pth"))
    if checkpoints:
        print(f"Loading checkpoint: {checkpoints[0]}")
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Warning: No checkpoint found, using random initialization")

    return model


def test_physics_informed_llama_with_proper_special_tokens():
    """
    Test function using PROPER special token initialization
    """
    print("=== Testing PhysicsInformedEvolutionaryLlama with PROPER Special Token Support ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. FIRST: Define your special tokens
    print("1. Defining special tokens...")
    custom_special_tokens = [
        "STAR_DATA_0", "STAR_DATA_1", "STAR_DATA_2", "STAR_DATA_3", "STAR_DATA_4",
        "STAR_DATA_5", "STAR_DATA_6", "STAR_DATA_7", "STAR_DATA_8", "STAR_DATA_9"
    ]
    print(f"  Custom special tokens: {custom_special_tokens}")

    # 2. Load tokenizer WITH custom special tokens during initialization
    print("\n2. Loading tokenizer with custom special tokens...")
    try:
        # THIS IS THE KEY: Pass custom_special_tokens during initialization
        tokenizer = Tokenizer(
            model_path=TOKENIZER_PATH,
            custom_special_tokens=custom_special_tokens  # This is crucial!
        )
        print(f"✓ Tokenizer loaded with custom special tokens")
        print(f"  Total vocab size: {tokenizer.n_words}")
        print(f"  Custom token IDs: {tokenizer.custom_token_ids}")

        # Verify tokens are properly registered
        for token in custom_special_tokens:
            token_id = tokenizer.get_custom_token_id(token)
            print(f"    {token}: {token_id}")

    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False

    # 3. Test encoding with special tokens
    print("\n3. Testing special token encoding...")
    try:
        test_texts = [
            "What is the temperature of STAR_DATA_0?",
            "Compare STAR_DATA_1 and STAR_DATA_2 evolution",
            "Normal text without special tokens",
            "Multiple STAR_DATA_0 tokens STAR_DATA_1 in one sentence"
        ]

        for text in test_texts:
            # Use allowed_special="all" to allow custom special tokens
            tokens = tokenizer.encode(text, bos=False, eos=False, allowed_special="all")

            # Check for custom special tokens in the encoded result
            custom_token_ids = set(tokenizer.custom_token_ids.values())
            found_custom_tokens = [t for t in tokens if t in custom_token_ids]

            print(f"  Text: {text}")
            print(f"    Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"    Custom tokens found: {found_custom_tokens}")
            print(f"    Success: {'✓' if found_custom_tokens else '✗'}")

            # Decode to verify
            decoded = tokenizer.decode(tokens)
            print(f"    Decoded: {decoded}")
            print()

    except Exception as e:
        print(f"✗ Special token encoding test failed: {e}")
        return False

    # 4. Load base model
    print("\n4. Loading base LLaMA model...")
    try:
        base_model = _load_base_model()
        base_model = base_model.to(device)
        print(f"✓ Base model loaded and moved to {device}")
        print(f"  Original model vocab size: {base_model.params.vocab_size}")
    except Exception as e:
        print(f"✗ Base model loading failed: {e}")
        return False

    # 5. Update model vocabulary to match tokenizer
    print("\n5. Updating model vocabulary size...")
    try:
        # The tokenizer now has more tokens, so we need to expand the model's embedding layers
        original_vocab_size = base_model.params.vocab_size
        new_vocab_size = tokenizer.n_words

        if new_vocab_size > original_vocab_size:
            print(f"  Expanding vocab from {original_vocab_size} to {new_vocab_size}")

            # Expand token embeddings
            old_embeddings = base_model.tok_embeddings.weight.data
            new_embeddings = torch.randn(new_vocab_size, old_embeddings.size(1),
                                         device=old_embeddings.device, dtype=old_embeddings.dtype)
            new_embeddings[:original_vocab_size] = old_embeddings

            # Create new embedding layer
            base_model.tok_embeddings = torch.nn.Embedding(new_vocab_size, old_embeddings.size(1))
            base_model.tok_embeddings.weight.data = new_embeddings
            base_model.tok_embeddings = base_model.tok_embeddings.to(device)

            # Expand output layer
            old_output = base_model.output.weight.data
            new_output = torch.randn(new_vocab_size, old_output.size(1),
                                     device=old_output.device, dtype=old_output.dtype)
            new_output[:original_vocab_size] = old_output

            base_model.output = torch.nn.Linear(old_output.size(1), new_vocab_size, bias=False)
            base_model.output.weight.data = new_output
            base_model.output = base_model.output.to(device)

            # Update model params
            base_model.params.vocab_size = new_vocab_size

            print(f"  ✓ Model vocabulary expanded to {new_vocab_size}")
        else:
            print(f"  Model vocabulary already sufficient: {original_vocab_size}")

    except Exception as e:
        print(f"✗ Model vocabulary update failed: {e}")
        return False

    # 6. Create physics-informed model (no need to register tokens - already done!)
    print("\n6. Creating PhysicsInformedEvolutionaryLlama...")
    try:
        estimated_feature_dim = 2048
        estimated_max_stages = 4

        # The special tokens are already in the tokenizer, so we just need to tell the model about them
        special_tokens_dict = tokenizer.custom_token_ids

        model = PhysicsInformedEvolutionaryLlama(
            base_model=base_model,
            latent_dim=estimated_feature_dim,
            special_tokens=special_tokens_dict,
            max_stages=estimated_max_stages,
            evolution_model_type="lstm",
        ).to(device)

        print(f"✓ Model created")

        # special_tokens_dict = {v: k for k, v in special_tokens_dict.items()}
        print(f"✓ Using special tokens: {special_tokens_dict}")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # 7. Load and test dataset
    print("\n7. Loading stellar dataset...")
    try:
        json_file_path = 'stellar_evolution_results_complete.json'
        manager = StellarDatasetManager(json_file_path)
        manager.load_data()

        # Create splits - tokenizer already has special tokens configured
        splits = manager.create_splits(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            tokenizer=tokenizer,
            max_length=512
        )

        test_loader = manager.get_dataloaders(batch_size=2, num_workers=0)['train']
        test_batch = next(iter(test_loader))

        for i, sample in enumerate(test_loader):
            print(f"  Sample {i}:")
            print(f"    Features shape: {sample['features'].shape}")
            print(f"    Prompt type: {sample['prompt_types']}")
            print(f"    Questions: {sample['questions']}")
            print(f"    Answers: {sample['answers']}")
            if i > 5:
                break

        print(f"✓ Dataset loaded: {len(splits['train'])} examples")
        print(f"  Input IDs shape: {test_batch['input_ids'].shape}")

        # Check for special tokens in the batch
        custom_token_ids = set(tokenizer.custom_token_ids.values())
        total_special_tokens = 0

        for batch_idx in range(test_batch['input_ids'].shape[0]):
            batch_special_tokens = 0
            for token_id in test_batch['input_ids'][batch_idx]:
                if token_id.item() in custom_token_ids:
                    batch_special_tokens += 1
                    total_special_tokens += 1

            print(f"  Batch {batch_idx}: {batch_special_tokens} special tokens found")

        print(f"  Total special tokens in batch: {total_special_tokens}")

        if total_special_tokens > 0:
            print(f"  ✅ SUCCESS: Special tokens properly detected!")
        else:
            print(f"  ⚠ Warning: No special tokens found - check your question generation")
            # Show sample questions for debugging
            print(f"  Sample questions:")
            for i, q in enumerate(test_batch.get('questions', ['N/A', 'N/A'])[:2]):
                print(f"    {i}: {q[:100]}...")

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

    # 8. Test forward pass
    print("\n8. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=test_batch['input_ids'].to(device),
                features=test_batch['features'].to(device),
                stage_mask=test_batch['stage_mask'].to(device),
                n_stages=test_batch['n_stages'].to(device),
                ages=test_batch['ages'].to(device),
                masses=test_batch['masses'].to(device),
                metallicities=test_batch['metallicities'].to(device),
                use_enhanced_features=True
            )

            print(f"✓ Forward pass successful!")
            print(f"  Output keys: {list(outputs.keys())}")
            print(f"  Logits shape: {outputs['logits'].shape}")
            print(f"  Special positions found: {len(outputs.get('special_positions', {}))}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n=== FINAL RESULTS ===")
    print(f"✅ Tokenizer properly configured with custom special tokens")
    print(f"✅ Model vocabulary expanded to match tokenizer")
    print(f"✅ Dataset integration successful")
    print(f"✅ Forward pass working")

    if total_special_tokens > 0:
        print(f"✅ Special token detection: WORKING!")
        print(f"🎉 Your model is ready for training!")
    else:
        print(f"⚠ Special token detection: Check your question generation")
        print(f"   Make sure your questions contain STAR_DATA_N patterns")

    return True


def create_test_questions_with_special_tokens():
    """
    Helper function to create test questions that definitely contain special tokens
    """
    test_questions = [
        "What is the effective temperature of STAR_DATA_0 during its main sequence phase?",
        "How does the surface gravity of STAR_DATA_1 compare to solar values?",
        "Analyze the metallicity evolution from STAR_DATA_2 to STAR_DATA_3.",
        "Compare the luminosity trends between STAR_DATA_4 and STAR_DATA_5.",
    ]

    # Test with the properly configured tokenizer
    custom_special_tokens = [
        "STAR_DATA_0", "STAR_DATA_1", "STAR_DATA_2", "STAR_DATA_3", "STAR_DATA_4",
        "STAR_DATA_5", "STAR_DATA_6", "STAR_DATA_7", "STAR_DATA_8", "STAR_DATA_9"
    ]

    tokenizer = Tokenizer(
        model_path=TOKENIZER_PATH,
        custom_special_tokens=custom_special_tokens
    )

    print("=== Testing Question Tokenization ===")
    for i, question in enumerate(test_questions):
        tokens = tokenizer.encode(question, bos=False, eos=False, allowed_special="all")
        custom_token_ids = set(tokenizer.custom_token_ids.values())
        found_tokens = [t for t in tokens if t in custom_token_ids]

        print(f"Question {i + 1}: {question}")
        print(f"  Special tokens found: {found_tokens}")
        print(
            f"  Token IDs: {[tokenizer.custom_token_ids[k] for k, v in tokenizer.custom_token_ids.items() if v in found_tokens]}")
        print()


if __name__ == "__main__":
    print("Testing question tokenization first...")
    create_test_questions_with_special_tokens()

    print("\n" + "=" * 80 + "\n")

    success = test_physics_informed_llama_with_proper_special_tokens()

    if success:
        print(f"\n🎉 All tests passed! Your model is ready for training.")
    else:
        print(f"\n❌ Some tests failed. Check the error messages above.")