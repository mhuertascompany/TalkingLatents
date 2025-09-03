import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist
from sklearn.model_selection import train_test_split
import os
from typing import Dict, List, Tuple, Optional, Union
import pickle


class StellarEvolutionDataset(Dataset):
    """
    Updated dataset with proper attention mask support
    """

    def __init__(self, data_list: List[Dict], split: str = 'train', tokenizer=None, max_length: int = 512):
        self.data = data_list
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Determine padding token ID
        self.pad_token_id = self._get_pad_token_id()
        self.eos_token_id = self._get_eos_token_id()
        self.bos_token_id = self._get_bos_token_id()
        
        print(f"Dataset initialized with:")
        print(f"  Padding token ID: {self.pad_token_id}")
        print(f"  EOS token ID: {self.eos_token_id}")
        print(f"  BOS token ID: {self.bos_token_id}")

        # Count examples
        self.single_stage_count = 0
        self.multi_stage_count = 0
        
        for item in self.data:
            n_stages = item.get('n_stages', 1)
            if n_stages == 1:
                self.single_stage_count += 1
            else:
                self.multi_stage_count += 1

        print(f"{split.upper()} SET: {len(self.data)} total examples")
        print(f"  - Single-stage (n_stages=1): {self.single_stage_count}")
        print(f"  - Multi-stage (n_stages>1): {self.multi_stage_count}")

    def _get_pad_token_id(self):
        """Determine the padding token ID"""
        if self.tokenizer is None:
            return 0
        
        # Try common padding token patterns
        padding_candidates = [
            '!',  # Your current problematic token
            '<|pad|>', 
            '<pad>', 
            '[PAD]',
            '<unk>'
        ]
        
        for candidate in padding_candidates:
            try:
                tokens = self.tokenizer.encode(candidate, bos=False, eos=False)
                if len(tokens) == 1:
                    print(f"Found padding token candidate '{candidate}' -> ID: {tokens[0]}")
                    return tokens[0]
            except:
                continue
        
        # If no specific padding token found, use 0
        print("Warning: No padding token found, using 0")
        return 0
    
    def _get_eos_token_id(self):
        """Determine the EOS token ID"""
        if self.tokenizer is None:
            return 1  # Common default
        
        eos_candidates = ['<|end_of_text|>', '</s>', '<|endoftext|>', '<eos>']
        
        for candidate in eos_candidates:
            try:
                tokens = self.tokenizer.encode(candidate, bos=False, eos=False)
                if len(tokens) == 1:
                    print(f"Found EOS token '{candidate}' -> ID: {tokens[0]}")
                    return tokens[0]
            except:
                continue
        
        print("Warning: No EOS token found, using 1")
        return 1
    
    def _get_bos_token_id(self):
        """Determine the BOS token ID"""
        if self.tokenizer is None:
            return 2  # Common default
        
        bos_candidates = ['<|begin_of_text|>', '<s>', '<|startoftext|>', '<bos>']
        
        for candidate in bos_candidates:
            try:
                tokens = self.tokenizer.encode(candidate, bos=False, eos=False)
                if len(tokens) == 1:
                    print(f"Found BOS token '{candidate}' -> ID: {tokens[0]}")
                    return tokens[0]
            except:
                continue
        
        print("Warning: No BOS token found, using 2")
        return 2

    def get_token_ids_with_attention(self, sequence: str, token_type='question') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text to token IDs and create attention mask
        
        Returns:
            token_ids: (max_length,) tensor of token IDs
            attention_mask: (max_length,) tensor of 0s and 1s (1 = real token, 0 = padding)
        """
        if self.tokenizer is None:
            # Create dummy data for testing
            dummy_length = min(20, self.max_length)
            token_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            
            # Fill with some dummy tokens
            token_ids[0] = self.bos_token_id
            token_ids[1:dummy_length-1] = torch.randint(100, 1000, (dummy_length-2,))  # Random content
            token_ids[dummy_length-1] = self.eos_token_id
            
            # Set attention mask
            attention_mask[:dummy_length] = 1
            
            return token_ids, attention_mask
        
        try:
            # Tokenize with BOS and EOS
            if token_type == 'question':
                bos, eos =True, False
            else:
                bos, eos = False, True
            tokens = self.tokenizer.encode(sequence, bos=bos, eos=eos, allowed_special="all")
            original_length = len(tokens)
            
            # Create attention mask for original tokens
            if original_length <= self.max_length:
                # Pad to max_length
                padded_tokens = tokens + [self.pad_token_id] * (self.max_length - original_length)
                attention_mask = [1] * original_length + [0] * (self.max_length - original_length)
            else:
                # Truncate to max_length
                padded_tokens = tokens[:self.max_length]
                attention_mask = [1] * self.max_length
                
                # Ensure EOS token at the end if truncated
                if padded_tokens[-1] != self.eos_token_id:
                    padded_tokens[-1] = self.eos_token_id
            
            return torch.tensor(padded_tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
            
        except Exception as e:
            print(f"Tokenization error for sequence: {sequence[:50]}... Error: {e}")
            # Fallback to padding
            token_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            return token_ids, attention_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item with attention masks"""
        item = self.data[idx]

        # Extract and validate features (unchanged)
        features = item.get('features', [])
        if isinstance(features, list):
            features = np.array(features)

        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            actual_stages = 1
        elif len(features.shape) == 2:
            actual_stages = features.shape[0]
        else:
            raise ValueError(f"Invalid features shape at idx {idx}: {features.shape}")

        features_tensor = torch.FloatTensor(features)

        # Extract question and answer (unchanged)
        result = item.get('result', {})
        if isinstance(result, dict):
            question = result.get('Question', '')
            answer = result.get('Answer', '')
        else:
            question = 'Describe this star based on the provided data.'
            answer = str(result) if result else ''

        # Tokenize with attention masks
        input_ids, input_attention_mask = self.get_token_ids_with_attention(question, token_type='question')
        output_ids, output_attention_mask = self.get_token_ids_with_attention(answer,  token_type='answer')

        # Extract physical parameters (unchanged)
        information = item.get('information', {})
        evolutionary_stages = item.get('evolutionary_stages', [])

        # Build stellar parameter arrays (unchanged)
        ages = []
        masses = []
        metallicities = []

        if len(evolutionary_stages) >= actual_stages:
            for i in range(actual_stages):
                stage = evolutionary_stages[i]
                ages.append(float(stage.get('age', 0.0)))
                masses.append(float(information.get('initial_mass', 
                                   information.get('mass', 1.0))))
                metallicities.append(float(information.get('metallicity', 
                                          information.get('feh', 0.0))))
        else:
            base_age = float(information.get('age', 0.0))
            base_mass = float(information.get('initial_mass', 
                            information.get('mass', 1.0)))
            base_metallicity = float(information.get('metallicity', 
                                   information.get('feh', 0.0)))
            
            ages = [base_age] * actual_stages
            masses = [base_mass] * actual_stages
            metallicities = [base_metallicity] * actual_stages

        ages_tensor = torch.FloatTensor(ages)
        masses_tensor = torch.FloatTensor(masses)
        metallicities_tensor = torch.FloatTensor(metallicities)

        # Create metadata (unchanged)
        metadata = {
            'track_id': item.get('track_id'),
            'star_id': item.get('star_id'),
            'information': information,
            'stage_mapping': item.get('stage_mapping', []),
            'evolutionary_stages': evolutionary_stages,
        }

        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'input_attention_mask': input_attention_mask,  # NEW
            'output_attention_mask': output_attention_mask,  # NEW
            'features': features_tensor,
            'question': question,
            'answer': answer,
            'metadata': metadata,
            'n_stages': actual_stages,
            'ages': ages_tensor,
            'masses': masses_tensor,
            'metallicities': metallicities_tensor,
            'idx': idx,
            'special_token_ids': {  # NEW: For generation
                'pad_token_id': self.pad_token_id,
                'eos_token_id': self.eos_token_id,
                'bos_token_id': self.bos_token_id
            }
        }

def enhanced_collate_fn(batch):
    """
    Enhanced collate function that properly handles attention masks
    """
    batch_size = len(batch)
    
    # Get special token IDs from first item
    special_tokens = batch[0]['special_token_ids']
    pad_token_id = special_tokens['pad_token_id']
    
    # Find maximum stages and feature dimension
    max_stages = max(item['n_stages'] for item in batch)
    feature_dim = batch[0]['features'].shape[-1]
    
    # Check if we have tokenized input
    has_input_ids = batch[0]['input_ids'] is not None
    if has_input_ids:
        seq_length = batch[0]['input_ids'].shape[0]

    # Initialize padded tensors for stellar data
    padded_features = torch.zeros(batch_size, max_stages, feature_dim)
    padded_ages = torch.zeros(batch_size, max_stages)
    padded_masses = torch.zeros(batch_size, max_stages)
    padded_metallicities = torch.zeros(batch_size, max_stages)
    stage_mask = torch.zeros(batch_size, max_stages, dtype=torch.bool)

    # Initialize padded tensors for text data with CORRECT padding token
    if has_input_ids:
        input_ids_batch = torch.full((batch_size, seq_length), pad_token_id, dtype=torch.long)
        output_ids_batch = torch.full((batch_size, seq_length), pad_token_id, dtype=torch.long)
        input_attention_mask_batch = torch.zeros(batch_size, seq_length, dtype=torch.long)
        output_attention_mask_batch = torch.zeros(batch_size, seq_length, dtype=torch.long)

    # Collect other data
    questions = []
    answers = []
    metadata = []
    n_stages_list = []
    indices = []

    # Fill tensors
    for i, item in enumerate(batch):
        n_stages = item['n_stages']
        
        # Stellar features and parameters
        padded_features[i, :n_stages, :] = item['features']
        padded_ages[i, :n_stages] = item['ages']
        padded_masses[i, :n_stages] = item['masses']
        padded_metallicities[i, :n_stages] = item['metallicities']
        stage_mask[i, :n_stages] = True
        
        # Text data with attention masks
        if has_input_ids:
            input_ids_batch[i] = item['input_ids']
            output_ids_batch[i] = item['output_ids']
            input_attention_mask_batch[i] = item['input_attention_mask']
            output_attention_mask_batch[i] = item['output_attention_mask']
        
        # Collect remaining data
        questions.append(item['question'])
        answers.append(item['answer'])
        metadata.append(item['metadata'])
        n_stages_list.append(n_stages)
        indices.append(item['idx'])

    # Build result dictionary
    result = {
        'features': padded_features,
        'stage_mask': stage_mask,
        'n_stages': torch.tensor(n_stages_list, dtype=torch.long),
        'ages': padded_ages,
        'masses': padded_masses,
        'metallicities': padded_metallicities,
        'questions': questions,
        'answers': answers,
        'metadata': metadata,
        'indices': indices,
        'batch_info': {
            'batch_size': batch_size,
            'max_stages': max_stages,
            'feature_dim': feature_dim,
            'has_input_ids': has_input_ids,
            'special_tokens': special_tokens
        }
    }

    if has_input_ids:
        result.update({
            'input_ids': input_ids_batch,
            'output_ids': output_ids_batch,
            'input_attention_mask': input_attention_mask_batch,
            'output_attention_mask': output_attention_mask_batch
        })

    return result

class StellarDatasetManager:
    """
    Complete manager for stellar evolution datasets.
    Handles loading, validation, splitting, and serving data.
    """

    def __init__(self, json_file_path: str, device=None):
        """
        Initialize the dataset manager.

        Args:
            json_file_path: Path to JSON file with results
            device: Device for distributed training (optional)
        """
        self.json_file_path = json_file_path
        self.device = device
        self.raw_data = None
        self.metadata = None
        self.splits = {}
        self.validation_results = None

    def load_data(self):
        """Load and validate data from JSON file."""
        print(f"Loading data from {self.json_file_path}")

        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")

        # Handle different JSON structures
        if isinstance(data, dict):
            if 'results' in data:
                self.raw_data = data['results']
                self.metadata = data.get('metadata', {})
            elif isinstance(data.get('data'), list):
                self.raw_data = data['data']
                self.metadata = {k: v for k, v in data.items() if k != 'data'}
            else:
                # Assume the whole dict contains data entries
                self.raw_data = [v for k, v in data.items() if isinstance(v, dict)]
                self.metadata = {}
        elif isinstance(data, list):
            self.raw_data = data
            self.metadata = {}
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

        print(f"Loaded {len(self.raw_data)} raw examples")
        
        # Validate and filter data
        self._validate_and_filter_data()
        
        return self.raw_data

    def _validate_and_filter_data(self):
        """Validate data and filter out invalid examples."""
        print("Validating data...")
        
        valid_data = []
        issues = []
        
        for i, item in enumerate(self.raw_data):
            # Check essential fields
            if not isinstance(item, dict):
                issues.append(f"Example {i}: Not a dictionary")
                continue
                
            if 'result' not in item or item['result'] is None:
                issues.append(f"Example {i}: Missing or null result")
                continue
                
            if 'features' not in item:
                issues.append(f"Example {i}: Missing features")
                continue

            # Validate features
            try:
                features = np.array(item['features'])
                if features.size == 0:
                    issues.append(f"Example {i}: Empty features")
                    continue
            except Exception as e:
                issues.append(f"Example {i}: Invalid features - {e}")
                continue

            # Check result format
            result = item['result']
            if isinstance(result, dict):
                if 'Question' not in result or 'Answer' not in result:
                    issues.append(f"Example {i}: Missing Question or Answer")
                    continue
            elif not isinstance(result, str):
                issues.append(f"Example {i}: Invalid result format")
                continue

            valid_data.append(item)

        self.raw_data = valid_data
        
        print(f"Validation complete:")
        print(f"  - Valid examples: {len(valid_data)}")
        print(f"  - Invalid examples: {len(issues)}")
        
        if issues:
            print(f"  - First 5 issues:")
            for issue in issues[:5]:
                print(f"    {issue}")

        # Store validation results
        self.validation_results = {
            'total_checked': len(self.raw_data) + len(issues),
            'valid_count': len(valid_data),
            'invalid_count': len(issues),
            'issues': issues
        }

    def get_data_statistics(self):
        """Get comprehensive statistics about the dataset."""
        if self.raw_data is None:
            self.load_data()

        stats = {
            'total_examples': len(self.raw_data),
            'single_stage_count': 0,
            'multi_stage_count': 0,
            'n_stages_distribution': {},
            'feature_shapes': [],
            'question_lengths': [],
            'answer_lengths': [],
            'track_ids': set(),
            'ages_range': [float('inf'), float('-inf')],
            'masses_range': [float('inf'), float('-inf')],
            'metallicities_range': [float('inf'), float('-inf')]
        }

        for item in self.raw_data:
            # Count stages
            n_stages = item.get('n_stages', 1)
            features = np.array(item.get('features', []))
            
            if len(features.shape) == 1:
                actual_stages = 1
            elif len(features.shape) == 2:
                actual_stages = features.shape[0]
            else:
                actual_stages = 1

            if actual_stages == 1:
                stats['single_stage_count'] += 1
            else:
                stats['multi_stage_count'] += 1

            # N stages distribution
            stats['n_stages_distribution'][actual_stages] = \
                stats['n_stages_distribution'].get(actual_stages, 0) + 1

            # Feature shapes
            stats['feature_shapes'].append(features.shape)

            # Text lengths
            result = item.get('result', {})
            if isinstance(result, dict):
                question = result.get('Question', '')
                answer = result.get('Answer', '')
                stats['question_lengths'].append(len(question))
                stats['answer_lengths'].append(len(answer))

            # Track IDs
            track_id = item.get('track_id')
            if track_id is not None:
                stats['track_ids'].add(track_id)

            # Physical parameters
            information = item.get('information', {})
            evolutionary_stages = item.get('evolutionary_stages', [])

            if evolutionary_stages:
                for stage in evolutionary_stages:
                    age = stage.get('age', 0)
                    if age > 0:
                        stats['ages_range'][0] = min(stats['ages_range'][0], age)
                        stats['ages_range'][1] = max(stats['ages_range'][1], age)

            mass = information.get('initial_mass', information.get('mass'))
            if mass is not None:
                stats['masses_range'][0] = min(stats['masses_range'][0], mass)
                stats['masses_range'][1] = max(stats['masses_range'][1], mass)

            metallicity = information.get('metallicity', information.get('feh'))
            if metallicity is not None:
                stats['metallicities_range'][0] = min(stats['metallicities_range'][0], metallicity)
                stats['metallicities_range'][1] = max(stats['metallicities_range'][1], metallicity)

        # Process statistics
        stats['unique_tracks'] = len(stats['track_ids'])
        del stats['track_ids']  # Remove set for JSON serialization

        if stats['question_lengths']:
            stats['text_stats'] = {
                'question_length': {
                    'mean': np.mean(stats['question_lengths']),
                    'min': np.min(stats['question_lengths']),
                    'max': np.max(stats['question_lengths'])
                },
                'answer_length': {
                    'mean': np.mean(stats['answer_lengths']),
                    'min': np.min(stats['answer_lengths']),
                    'max': np.max(stats['answer_lengths'])
                }
            }

        stats['feature_stats'] = {
            'unique_shapes': list(set(stats['feature_shapes'])),
            'total_shapes': len(stats['feature_shapes'])
        }

        return stats

    def print_summary(self):
        """Print a comprehensive summary of the dataset."""
        stats = self.get_data_statistics()

        print("\n" + "="*60)
        print("STELLAR EVOLUTION DATASET SUMMARY")
        print("="*60)

        print(f"\nDataset Overview:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Single-stage: {stats['single_stage_count']}")
        print(f"  Multi-stage: {stats['multi_stage_count']}")
        print(f"  Unique tracks: {stats['unique_tracks']}")

        print(f"\nStages Distribution:")
        for n_stages, count in sorted(stats['n_stages_distribution'].items()):
            percentage = (count / stats['total_examples']) * 100
            print(f"  {n_stages} stage(s): {count} ({percentage:.1f}%)")

        print(f"\nFeature Information:")
        unique_shapes = stats['feature_stats']['unique_shapes']
        print(f"  Unique shapes: {len(unique_shapes)}")
        for shape in sorted(unique_shapes)[:5]:  # Show first 5
            print(f"    {shape}")
        if len(unique_shapes) > 5:
            print(f"    ... and {len(unique_shapes) - 5} more")

        if 'text_stats' in stats:
            ts = stats['text_stats']
            print(f"\nText Statistics:")
            print(f"  Question length: {ts['question_length']['mean']:.1f} chars "
                  f"({ts['question_length']['min']}-{ts['question_length']['max']})")
            print(f"  Answer length: {ts['answer_length']['mean']:.1f} chars "
                  f"({ts['answer_length']['min']}-{ts['answer_length']['max']})")

        print(f"\nPhysical Parameters:")
        if stats['ages_range'][0] != float('inf'):
            print(f"  Ages: {stats['ages_range'][0]:.2f} - {stats['ages_range'][1]:.2f} Gyr")
        if stats['masses_range'][0] != float('inf'):
            print(f"  Masses: {stats['masses_range'][0]:.2f} - {stats['masses_range'][1]:.2f} M☉")
        if stats['metallicities_range'][0] != float('inf'):
            print(f"  Metallicities: {stats['metallicities_range'][0]:.3f} - {stats['metallicities_range'][1]:.3f}")

        if self.validation_results:
            vr = self.validation_results
            print(f"\nData Quality:")
            print(f"  Valid examples: {vr['valid_count']}/{vr['total_checked']}")
            if vr['invalid_count'] > 0:
                print(f"  Filtered out: {vr['invalid_count']} invalid examples")

        print("="*60)

    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      stratify_by='n_stages', random_state=42, 
                      tokenizer=None, max_length=512):
        """
        Create train/validation/test splits.

        Args:
            train_ratio: Training set proportion
            val_ratio: Validation set proportion  
            test_ratio: Test set proportion
            stratify_by: Stratification strategy ('n_stages' or None)
            random_state: Random seed
            tokenizer: Text tokenizer
            max_length: Maximum token sequence length
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        if self.raw_data is None:
            self.load_data()

        data = self.raw_data.copy()

        # Prepare stratification
        if stratify_by == 'n_stages':
            # Use actual stages from features for stratification
            stratify_labels = []
            for item in data:
                features = np.array(item.get('features', []))
                if len(features.shape) == 1:
                    stratify_labels.append(1)
                elif len(features.shape) == 2:
                    stratify_labels.append(features.shape[0])
                else:
                    stratify_labels.append(1)
        else:
            stratify_labels = None

        # Create splits
        if val_ratio + test_ratio > 0:
            train_data, temp_data = train_test_split(
                data,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=stratify_labels
            )

            if val_ratio > 0 and test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                
                # Get stratify labels for temp data
                if stratify_labels:
                    temp_stratify = stratify_labels[len(train_data):]
                else:
                    temp_stratify = None
                
                val_data, test_data = train_test_split(
                    temp_data,
                    test_size=(1 - val_size),
                    random_state=random_state,
                    stratify=temp_stratify
                )
            elif val_ratio > 0:
                val_data = temp_data
                test_data = []
            else:
                val_data = []
                test_data = temp_data
        else:
            train_data = data
            val_data = []
            test_data = []

        # Create dataset objects
        self.splits['train'] = StellarEvolutionDataset(train_data, 'train', tokenizer, max_length)
        if val_data:
            self.splits['val'] = StellarEvolutionDataset(val_data, 'val', tokenizer, max_length)
        if test_data:
            self.splits['test'] = StellarEvolutionDataset(test_data, 'test', tokenizer, max_length)

        print(f"\nDataset splits created successfully!")
        return self.splits

    def get_dataloaders(self, batch_size=32, num_workers=4, shuffle_train=True, 
                        world_size=1, pin_memory=True):
        """Create DataLoaders with enhanced collate function"""
        if not self.splits:
            raise RuntimeError("No splits created. Call create_splits() first.")

        dataloaders = {}

        for split_name, dataset in self.splits.items():
            shuffle = shuffle_train if split_name == 'train' else False
            sampler = None

            if world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=self.device if isinstance(self.device, int) else 0
                )
                shuffle = False

            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=pin_memory,
                collate_fn=enhanced_collate_fn  # Use the enhanced collate function
            )

        return dataloaders

    def _collate_fn(self, batch):
        """
        Custom collate function for batching examples with different numbers of stages.
        """
        batch_size = len(batch)
        
        # Find maximum stages and feature dimension
        max_stages = max(item['n_stages'] for item in batch)
        feature_dim = batch[0]['features'].shape[-1]
        
        # Check if we have tokenized input
        has_input_ids = batch[0]['input_ids'] is not None
        if has_input_ids:
            seq_length = batch[0]['input_ids'].shape[0]

        # Initialize padded tensors
        padded_features = torch.zeros(batch_size, max_stages, feature_dim)
        padded_ages = torch.zeros(batch_size, max_stages)
        padded_masses = torch.zeros(batch_size, max_stages)
        padded_metallicities = torch.zeros(batch_size, max_stages)
        stage_mask = torch.zeros(batch_size, max_stages, dtype=torch.bool)

        if has_input_ids:
            input_ids_batch = torch.zeros(batch_size, seq_length, dtype=torch.long)
            output_ids_batch = torch.zeros(batch_size, seq_length, dtype=torch.long)

        # Collect other data
        questions = []
        answers = []
        metadata = []
        n_stages_list = []
        indices = []

        # Fill tensors
        for i, item in enumerate(batch):
            n_stages = item['n_stages']
            
            # Features
            padded_features[i, :n_stages, :] = item['features']
            
            # Physical parameters
            padded_ages[i, :n_stages] = item['ages']
            padded_masses[i, :n_stages] = item['masses']
            padded_metallicities[i, :n_stages] = item['metallicities']
            
            # Stage mask
            stage_mask[i, :n_stages] = True
            
            # Text data
            if has_input_ids:
                input_ids_batch[i] = item['input_ids']
                output_ids_batch[i] = item['output_ids']
            
            # Collect remaining data
            questions.append(item['question'])
            answers.append(item['answer'])
            metadata.append(item['metadata'])
            n_stages_list.append(n_stages)
            indices.append(item['idx'])

        # Build result dictionary
        result = {
            'features': padded_features,  # (batch_size, max_stages, feature_dim)
            'stage_mask': stage_mask,     # (batch_size, max_stages)
            'n_stages': torch.tensor(n_stages_list, dtype=torch.long),  # (batch_size,)
            'ages': padded_ages,          # (batch_size, max_stages)
            'masses': padded_masses,      # (batch_size, max_stages)
            'metallicities': padded_metallicities,  # (batch_size, max_stages)
            'questions': questions,
            'answers': answers,
            'metadata': metadata,
            'indices': indices,
            'batch_info': {
                'batch_size': batch_size,
                'max_stages': max_stages,
                'feature_dim': feature_dim,
                'has_input_ids': has_input_ids
            }
        }

        if has_input_ids:
            result['input_ids'] = input_ids_batch
            result['output_ids'] = output_ids_batch

        return result

    def save_splits(self, save_dir):
        """Save dataset splits to disk."""
        if not self.splits:
            raise RuntimeError("No splits to save. Create splits first.")

        os.makedirs(save_dir, exist_ok=True)

        # Save each split
        for split_name, dataset in self.splits.items():
            split_path = os.path.join(save_dir, f"{split_name}_data.pkl")
            with open(split_path, 'wb') as f:
                pickle.dump(dataset.data, f)
            print(f"Saved {split_name} split: {len(dataset)} examples")

        # Save metadata
        metadata = {
            'source_file': self.json_file_path,
            'original_metadata': self.metadata,
            'validation_results': self.validation_results,
            'statistics': self.get_data_statistics()
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(metadata), f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")

    def load_splits(self, save_dir, tokenizer=None, max_length=512):
        """Load dataset splits from disk."""
        self.splits = {}
        
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(save_dir, f"{split_name}_data.pkl")
            if os.path.exists(split_path):
                with open(split_path, 'rb') as f:
                    data = pickle.load(f)
                self.splits[split_name] = StellarEvolutionDataset(
                    data, split_name, tokenizer, max_length)
                print(f"Loaded {split_name}: {len(self.splits[split_name])} examples")
        
        # Load metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.metadata = metadata.get('original_metadata', {})
            self.validation_results = metadata.get('validation_results', {})
            print("Loaded metadata")
        
        return self.splits


# Utility functions
def test_dataset_manager(json_file_path, n_samples=3):
    """Test the dataset manager with a small sample."""
    print(f"Testing StellarDatasetManager with {json_file_path}")
    
    # Initialize and load
    manager = StellarDatasetManager(json_file_path)
    manager.load_data()
    manager.print_summary()
    
    # Create splits
    manager.create_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Test dataloaders
    dataloaders = manager.get_dataloaders(batch_size=4, num_workers=0)
    
    # Test a batch
    print(f"\nTesting batch loading...")
    for split_name, loader in dataloaders.items():
        batch = next(iter(loader))
        print(f"{split_name.upper()} batch:")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Stage mask shape: {batch['stage_mask'].shape}")
        print(f"  N stages: {batch['n_stages']}")
        print(f"  Batch info: {batch['batch_info']}")
        break  # Just test one batch
    
    # Test individual samples
    print(f"\nTesting individual samples:")
    dataset = manager.splits['train']
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: features={sample['features'].shape}, "
              f"n_stages={sample['n_stages']}, "
              f"question_len={len(sample['question'])}")
    
    print("Test completed successfully!")
    return manager


if __name__ == "__main__":
    # Test with your file
    json_file = "stellar_evolution_results_complete.json"
    manager = test_dataset_manager(json_file, n_samples=5)