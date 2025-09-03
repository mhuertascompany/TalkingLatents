import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import json
import os
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from functools import partial
from google import genai
from itertools import chain
import re
from typing import Dict, List, Union, Any
import sys

warnings.filterwarnings('ignore')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

# Import your existing functions (assuming they're available)
from data.stellar_evolution import generate_evolutionary_tracks, tracks_to_dataframe


def merge_dataframes_and_filter_array(df1, df2, right_on, left_on, array):
    """
    Merge two dataframes on all columns and filter corresponding numpy array
    Parameters:
    - df1: First dataframe (berger_lamost)
    - df2: Second dataframe (features_meta)
    - array: Numpy array aligned with df2 (features)
    Returns:
    - merged_df: Merged dataframe
    - filtered_array: Filtered numpy array
    """
    # Track original indices
    df2_indexed = df2.reset_index().rename(columns={'index': 'original_idx'})
    # Merge on all original columns from df2
    merged = pd.merge(df1, df2_indexed, right_on=right_on, left_on=left_on, how='inner', suffixes=['', '_2'])
    # Filter array based on surviving indices
    surviving_indices = merged['original_idx'].values
    filtered_array = array[surviving_indices]
    # Clean up merged dataframe
    final_df = merged.drop('original_idx', axis=1).reset_index(drop=True)
    return final_df, filtered_array


def giant_cond(teff, logg):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :return: boolean
    """
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh


def parse_gemini_json_response(response_text):
    """
    Parse JSON response from Gemini, handling various markdown formats

    Args:
        response_text (str): Raw response text from Gemini

    Returns:
        dict or None: Parsed JSON dictionary, or None if parsing fails
    """
    import json
    import re

    # Clean the response text
    text = response_text.strip()

    # Remove markdown code blocks - handle various formats
    patterns = [
        r'^```json\s*(.*?)\s*```$',  # ```json ... ```
        r'^```\s*(.*?)\s*```$',  # ``` ... ```
        r'^`(.*?)`$',  # `...`
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            break

    # Try to parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Text being parsed: {text[:200]}...")
        return None


# Your existing Gemini setup
def init_gemini_client():
    """Initialize Gemini client using your existing setup"""
    try:
        with open('google_api.txt', 'r') as f:
            google_api_key = f.read().strip()
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    except FileNotFoundError:
        print("Warning: google_api.txt not found. Gemini client not initialized.")
        return None


# Initialize client globally as in your code
try:
    with open('google_api.txt', 'r') as f:
        google_api_key = f.read()
    os.environ["GOOGLE_API_KEY"] = google_api_key
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
except:
    client = None
    print("Warning: Could not initialize Gemini client")


def create_star_prompt(example):
    """
    Create a diverse prompt that GUARANTEES STAR_DATA_0 inclusion through post-processing
    Modified to handle single-stage as multi-stage with one stage
    """
    # Check if this is a single stage being treated as multi-stage
    evolutionary_stages = example.get('evolutionary_stages', [])
    if not evolutionary_stages:
        # Convert single stage to multi-stage format
        evolutionary_stages = [{
            'age': example.get('age', 0),
            'Teff': example.get('Teff', 0),
            'logg': example.get('logg', 0),
            'L': example.get('L', 0),
            'evolutionary_phase': example.get('evolutionary_phase', 'unknown')
        }]

    n_stages = len(evolutionary_stages)

    # Base prompt - let LLM be creative but guide it
    base_prompt = """
Describe this star in detail. Act like a scientist. Do not converse please, just give your description in a scientific tone.

Here is additional information about this star:
"""

    # Helper function to format values appropriately
    def format_value(key, value):
        # Skip None values, PIL image objects, and index values
        if value is None or str(type(value)).find('PIL') >= 0 or key == '__index_level_0__':
            return None

        # Handle numeric values
        if isinstance(value, (int, float)):
            # Format temperatures
            if key in ['Teff', 'temperature']:
                return f"{value:.0f}"
            # Format surface gravity
            elif key in ['logg', 'surface_gravity']:
                return f"{value:.2f}"
            # Format metallicity
            elif key in ['feh', 'metallicity', 'FeH']:
                return f"{value:.2f}"
            # Format masses
            elif key in ['mass', 'Mstar', 'initial_mass']:
                return f"{value:.2f}"
            # Format luminosity
            elif key in ['Lstar', 'luminosity', 'L']:
                return f"{value:.3f}"
            # Format age
            elif key in ['age']:
                return f"{value:.2f}"
            # Format coordinates
            elif key in ['_RA', '_DE']:
                return f"{value:.6f}"
            # Format error values
            elif key.find('err') > 0 or key.startswith('e_') or key.startswith('sig'):
                return f"{value:.6f}"
            # Default number formatting
            return f"{value:.4f}" if isinstance(value, float) else f"{value}"

        # String values
        return value

    # Group metadata into categories
    categories = {
        "Basic Information": [
            'KID', 'star_id', '_RA', '_DE', 'track_id'
        ],
        "Stellar Parameters": [
            'Teff', 'logg', 'FeH', 'feh', 'metallicity'
        ],
        "Physical Properties": [
            'Mstar', 'mass', 'initial_mass', 'Lstar', 'luminosity', 'L'
        ],
        "Evolution Properties": [
            'age', 'Main-Sequence', 'Giant', 'main_sequence', 'giant', 'evolutionary_phase'
        ],
    }

    # Build the prompt with organized scientific data
    formatted_prompt = base_prompt

    for category, keys in categories.items():
        # Check if we have any values for this category
        category_values = {}
        for key in keys:
            if key in example and example[key] is not None:
                formatted_value = format_value(key, example[key])
                if formatted_value is not None:
                    category_values[key] = formatted_value

        # If we have values, add the category
        if category_values:
            formatted_prompt += f"\n\n{category}:"
            for key, value in category_values.items():
                # Format key for better readability
                display_key = key.replace('_', ' ').replace('-', ' ')
                display_key = ' '.join(word.capitalize() for word in display_key.split())
                formatted_prompt += f"\n- {display_key}: {value}"

    # Add evolutionary stages information
    if n_stages > 1:
        formatted_prompt += f"\n\nEvolutionary Sequence ({n_stages} stages observed):\n"
        for i, stage in enumerate(evolutionary_stages):
            formatted_prompt += f"""
Stage {i} - Age {stage['age']:.2f} Gyr ({stage['evolutionary_phase']}):
- Effective Temperature: {stage['Teff']:.0f} K
- Surface Gravity: {stage['logg']:.2f} dex
- Luminosity: {stage['L']:.3f} L☉
"""
    else:
        stage = evolutionary_stages[0]
        formatted_prompt += f"\n\nSingle Observation:"
        formatted_prompt += f"\n- Age: {stage['age']:.2f} Gyr"
        formatted_prompt += f"\n- Evolutionary Phase: {stage['evolutionary_phase']}"

    # Create stage tokens
    stage_tokens = [f"STAR_DATA_{i}" for i in range(n_stages)]
    stage_list = ", ".join(stage_tokens)

    # Add diverse question generation instruction
    if n_stages > 1:
        formatted_prompt += f"""

Based on this evolutionary sequence, create a diverse, engaging question and provide a detailed scientific description.

Create a unique question about this star's evolution that:
- Asks about evolutionary processes, stellar physics, or temporal changes
- References the star's data using the placeholders: {stage_list}
- Could focus on any aspect: mass loss, nuclear burning, structural changes, HR diagram evolution, etc.
- Should sound like a real research question about stellar evolution
- Must include ALL stage placeholders: {stage_list}

Examples of good evolutionary question styles (but create your own unique variation):
- "How does the star evolve from {stage_tokens[0]} through {stage_tokens[-1]}, and what physical processes drive these changes?"
- "What can we learn about stellar nucleosynthesis by comparing {stage_tokens[0]} and {stage_tokens[-1]}?"

IMPORTANT: Keep your response concise and under 400 words total for both Question and Answer combined.
Your response must be ONLY valid JSON without any markdown formatting or code blocks. Return only the raw JSON object in exactly this structure:

{{
"Question": "Your diverse, engaging question about the evolutionary progression using all stage tokens {stage_list}...",
"Answer": "Detailed but concise scientific description of the star's evolution (keep under 300 words)..."
}}

Both Question and Answer fields must contain exactly these tokens: {stage_list} - all of them, no more, no less.
Both Question and Answer fields must not contain any information about the evolution track id of the star. both track_<id>_age_<age> and explicit track id should not be included 
"""
    else:
        formatted_prompt += f"""

Based on this information and the stellar data, provide a detailed scientific description of this star. 

Create a diverse, engaging question about this star that:
- Asks about stellar properties, evolution, or characteristics
- Uses scientific terminology appropriately  
- Includes the placeholder STAR_DATA_0 to reference the star's numerical data
- Could be about temperature, mass, luminosity, evolutionary stage, or any stellar property
- Should sound like a real astronomy question from a student or researcher

Examples of good question styles (but create your own unique variation):
- "What can we learn about the evolutionary stage of STAR_DATA_0?"
- "How do the stellar parameters of STAR_DATA_0 compare to typical main sequence stars?"

IMPORTANT: Keep your response concise and under 400 words total for both Question and Answer combined.
Your response must be ONLY valid JSON without any markdown formatting or code blocks. Return only the raw JSON object in exactly this structure:

{{
"Question": "Your diverse, engaging question about STAR_DATA_0...",
"Answer": "Detailed but concise scientific description of the star (keep under 300 words)..."
}}

The Question field must contain exactly one instance of "STAR_DATA_0" - no more, no less.
Both Question and Answer fields must not contain any information about the evolution track id of the star. both track_<id>_age_<age> and explicit track id should not be included 

"""

    return formatted_prompt


def create_evolutionary_star_prompt(example):
    """
    Create diverse evolutionary prompts with guaranteed proper special tokens
    Modified to use the unified prompt structure
    """
    return create_star_prompt(example)


def create_stellar_conversation_prompt(example):
    """
    Your existing function adapted for stellar evolution.
    Generate a prompt for creating a synthetic conversation about a star
    between a curious human and an astronomical AI assistant.
    Modified to handle both single and multi-stage uniformly.

    Args:
        example: Dictionary containing stellar metadata

    Returns:
        str: Formatted prompt to generate an entertaining educational conversation
    """
    # Check if this is a single stage being treated as multi-stage
    evolutionary_stages = example.get('evolutionary_stages', [])
    if not evolutionary_stages:
        # Convert single stage to multi-stage format
        evolutionary_stages = [{
            'age': example.get('age', 0),
            'Teff': example.get('Teff', 0),
            'logg': example.get('logg', 0),
            'L': example.get('L', 0),
            'evolutionary_phase': example.get('evolutionary_phase', 'unknown')
        }]

    n_stages = len(evolutionary_stages)
    stage_tokens = [f"STAR_DATA_{i}" for i in range(n_stages)]
    stage_list = ", ".join(stage_tokens)

    # First get all the organized stellar data using our existing function
    star_data = create_star_prompt(example).split(
        "Here is additional information about this star:\n\n")[1].split(
        "\n\nBased on this information")[0]

    conversation_prompt = f"""
Generate an entertaining and educational conversation between a curious human and an astronomical AI assistant passionate about stars and stellar evolution. The conversation should use the following stellar data as a rough guide (but not reference it directly):
{star_data}

The conversation should follow this pattern:

1. Human asks about the star {'and its evolutionary stages' if n_stages > 1 else 'and its properties'}
2. AI explains the basic features enthusiastically, using a friendly tone with occasional astronomy puns
3. Human asks a follow-up question about something specific (evolutionary phase, stellar mass, temperature, etc.)
4. AI provides a more detailed explanation, connecting the feature to broader stellar evolution concepts
5. Human expresses amazement and asks another question
6. AI shares an interesting fact or comparison about this type of star or evolutionary stage

Guidelines for AI's personality:
- Enthusiastic and passionate about stellar astronomy and evolution
- Uses accessible language but includes proper scientific terminology
- Occasionally uses space-related puns or expressions ("out of this world," "stellar example," etc.)
- Anthropomorphizes stars occasionally ("this star seems to be having a mid-life crisis")
- Connects observations to broader astronomical concepts
- Expresses wonder at stellar evolution and cosmic beauty

Both Dr. Stella and the student must include the tokens {stage_list} when they refer to the data of the star
(for example "I'm curious about this star: {stage_tokens[0]}" {'and "How does it evolve to " + stage_tokens[-1]' if n_stages > 1 else ''}). This will be used
as placeholders for numerical data about the star.

IMPORTANT: Keep the conversation concise - aim for 6-8 total exchanges with each response under 100 words.

Your response must be valid JSON following this structure:
{{
  "conversation": [
    {{
      "speaker": "human",
      "text": "Question about the star..."
    }},
    {{
      "speaker": "AI",
      "text": "Enthusiastic, educational response about the star..."
    }},
    // Additional turns following the pattern above (keep total under 8 exchanges)
  ]
}}

The conversation should be informative while remaining scientifically accurate about stellar evolution. 
Do not include references to the values above, simply use them to roughly guide your conversation.
"""
    return conversation_prompt


def create_evolutionary_conversation_prompt(example):
    """
    Generate a conversation prompt for stellar evolution across multiple stages.
    Modified to use the unified conversation structure.

    Args:
        example: Dictionary containing stellar metadata with multiple stages

    Returns:
        str: Formatted prompt for evolutionary conversation
    """
    return create_stellar_conversation_prompt(example)


def caption_star(features, information, conversation=False):
    """
    Enhanced caption function with improved JSON parsing and length limits
    Uses Gemini's native JSON mode for better reliability
    """
    global client
    if client is None:
        print("Warning: Gemini client not available")
        return None

    # Determine prompt type and create appropriate prompt
    prompt_type = information.get('prompt_type', 'single_stage')

    if conversation:
        prompt = create_stellar_conversation_prompt(information)
        if isinstance(features, np.ndarray) and len(features.shape) > 1:
            features_text = f"Spectral features for {len(features)} evolutionary stages"
        else:
            features_text = f"Spectral features: {features[:10].tolist() if hasattr(features, 'tolist') else features}"
    else:
        prompt = create_star_prompt(information)
        if isinstance(features, np.ndarray):
            if len(features.shape) > 1:
                features_text = f"Spectral features for {len(features)} evolutionary stages"
            else:
                features_text = f"Spectral features (first 10): {features[:10].tolist()}"
        else:
            features_text = f"Features: {features}"

    # Generate response from Gemini with JSON mode
    try:
        if conversation:
            # For conversation mode, try with JSON mode first
            response = client.models.generate_content(
                contents=[prompt, features_text],
                model="gemini-2.0-flash",
                config={
                    "response_mime_type": "application/json"
                }
            )
        else:
            # For Q&A mode, use JSON mode
            response = client.models.generate_content(
                contents=[prompt, features_text],
                model="gemini-2.0-flash",
                config={
                    "response_mime_type": "application/json"
                }
            )
    except Exception as e:
        print(f"Error generating content with JSON mode: {e}")
        # Fallback to regular mode
        try:
            response = client.models.generate_content(
                contents=[prompt, features_text],
                model="gemini-2.0-flash",
            )
        except Exception as e2:
            print(f"Error generating content: {e2}")
            return None

    if conversation:
        # For conversation mode, try multiple times to get valid JSON
        for attempt in range(5):  # Reduced attempts since JSON mode is more reliable
            if attempt > 0:
                print(f"Attempting try {attempt + 1}/5 for stellar conversation")

            try:
                response_text = response.text
                conversation_data = parse_gemini_json_response(response_text)

                if conversation_data is not None:
                    return conversation_data

            except Exception as e:
                print(f"Error parsing response: {e}")

            # Retry with new request
            try:
                response = client.models.generate_content(
                    contents=[prompt, features_text],
                    model="gemini-2.0-flash",
                    config={
                        "response_mime_type": "application/json"
                    }
                )
            except Exception as e:
                print(f"Error on retry: {e}")
                return None

        return None
    else:
        # For non-conversation mode, return parsed JSON
        try:
            response_text = response.text
            parsed_result = parse_gemini_json_response(response_text)

            if parsed_result is not None:
                return parsed_result
            else:
                # Fallback: return raw text if JSON parsing fails
                print("Warning: JSON parsing failed, returning raw text")
                return response_text

        except Exception as e:
            print(f"Error getting response text: {e}")
            return None


def process_example(example_data, conversation=False):
    """
    Enhanced processing function that handles both single-stage and evolutionary examples
    Modified to ensure uniform output structure
    """
    try:
        features = example_data['features']
        information = example_data['information']

        # Ensure we have evolutionary_stages and n_stages for all examples
        if 'evolutionary_stages' not in information:
            # Convert single stage to multi-stage format
            information['evolutionary_stages'] = [{
                'age': information.get('age', 0),
                'Teff': information.get('Teff', 0),
                'logg': information.get('logg', 0),
                'L': information.get('L', 0),
                'evolutionary_phase': information.get('evolutionary_phase', 'unknown')
            }]
            information['n_stages'] = 1
            information['stage_mapping'] = [{
                'stage_index': 0,
                'age': information.get('age', 0),
                'real_star_idx': information.get('real_star_idx', 0)
            }]

        # Ensure features are in the right format
        if information['n_stages'] == 1 and isinstance(features, np.ndarray) and len(features.shape) == 1:
            # Convert 1D features to 2D for consistency
            features = features.reshape(1, -1)

        result = caption_star(features, information, conversation)

        # Create return structure - now uniform for all examples
        return_data = {
            'track_id': information.get('track_id', 'unknown'),
            'star_id': information.get('star_id', 'unknown'),
            'n_stages': information.get('n_stages', 1),
            'evolutionary_stages': information.get('evolutionary_stages', []),
            'stage_mapping': information.get('stage_mapping', []),
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'result': result,
            'information': information
        }

        return return_data

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def replace_track_ids_with_tokens(text: str, information: Dict[str, Any]) -> str:
    """
    Replace track ID references with STAR_DATA_N tokens based on evolutionary stage.
    Modified to handle the unified structure.

    Parameters:
    -----------
    text : str
        The text (question or answer) that may contain track ID references
    information : dict
        Information dictionary containing n_stages and other metadata

    Returns:
    --------
    str: Text with track IDs replaced by appropriate STAR_DATA_N tokens
    """
    if not isinstance(text, str):
        return text

    n_stages = information.get('n_stages', 1)

    if n_stages == 1:
        # For single stage, just replace any track references with STAR_DATA_0
        # Pattern matches: track_<id>_age_<age> or track_<id> or similar variations
        patterns = [
            r'track_\d+_age_[\d.]+',  # track_0_age_4.61
            r'track_\d+',  # track_0
            r'Track\s+\d+',  # Track 0
            r'track\s+\d+',  # track 0
        ]

        modified_text = text
        for pattern in patterns:
            modified_text = re.sub(pattern, 'STAR_DATA_0', modified_text, flags=re.IGNORECASE)

        return modified_text

    else:
        # For multi-stage prompts, we need to map different stages to different tokens
        modified_text = text

        # Get stage mapping if available
        stage_mapping = information.get('stage_mapping', [])
        track_id = information.get('track_id', None)

        if stage_mapping and track_id is not None:
            # Create mapping from age to stage index
            age_to_stage = {}
            for stage_info in stage_mapping:
                age = stage_info['age']
                stage_idx = stage_info['stage_index']
                age_to_stage[age] = stage_idx

            # Sort ages to create ordered mapping
            sorted_ages = sorted(age_to_stage.keys())

            # Replace specific track_id_age_X.XX patterns with appropriate STAR_DATA_N
            track_age_pattern = rf'track_{track_id}_age_([\d.]+)'

            def replace_with_stage_token(match):
                age_str = match.group(1)
                try:
                    age = float(age_str)
                    # Find the closest age in our mapping
                    closest_age = min(sorted_ages, key=lambda x: abs(x - age))
                    stage_idx = age_to_stage[closest_age]
                    return f'STAR_DATA_{stage_idx}'
                except (ValueError, KeyError):
                    # Fallback to STAR_DATA_0 if we can't match
                    return 'STAR_DATA_0'

            modified_text = re.sub(track_age_pattern, replace_with_stage_token, modified_text, flags=re.IGNORECASE)

        # Also handle general track references by replacing with all stage tokens
        general_track_patterns = [
            rf'track_{track_id}(?!_age)',  # track_0 but not track_0_age_X.XX
            rf'Track\s+{track_id}',  # Track 0
            rf'track\s+{track_id}',  # track 0
        ] if track_id is not None else []

        # Create comma-separated list of all stage tokens
        all_stage_tokens = ', '.join([f'STAR_DATA_{i}' for i in range(n_stages)])

        for pattern in general_track_patterns:
            modified_text = re.sub(pattern, all_stage_tokens, modified_text, flags=re.IGNORECASE)

        # Handle any remaining generic track patterns
        generic_patterns = [
            r'track_\d+_age_[\d.]+',  # Any remaining track_X_age_Y.YY
            r'track_\d+',  # Any remaining track_X
            r'Track\s+\d+',  # Any remaining Track X
            r'track\s+\d+',  # Any remaining track X
        ]

        for pattern in generic_patterns:
            # Replace with STAR_DATA_0 as fallback
            modified_text = re.sub(pattern, 'STAR_DATA_0', modified_text, flags=re.IGNORECASE)

    return modified_text


def clean_result_data(result_data: Union[Dict, str], information: Dict[str, Any]) -> Union[Dict, str]:
    """
    Clean result data by removing track ID references and replacing with appropriate tokens.

    Parameters:
    -----------
    result_data : dict or str
        The result data from Gemini (could be parsed JSON dict or raw string)
    information : dict
        Information dictionary containing n_stages and other metadata

    Returns:
    --------
    dict or str: Cleaned result data with track IDs replaced
    """
    if isinstance(result_data, dict):
        cleaned_result = {}

        # Handle different possible structures
        if 'Question' in result_data and 'Answer' in result_data:
            # Standard Q&A format
            cleaned_result['Question'] = replace_track_ids_with_tokens(
                result_data['Question'], information
            )
            cleaned_result['Answer'] = replace_track_ids_with_tokens(
                result_data['Answer'], information
            )

        elif 'conversation' in result_data:
            # Conversation format
            cleaned_conversation = []
            for turn in result_data['conversation']:
                cleaned_turn = turn.copy()
                if 'text' in cleaned_turn:
                    cleaned_turn['text'] = replace_track_ids_with_tokens(
                        cleaned_turn['text'], information
                    )
                cleaned_conversation.append(cleaned_turn)
            cleaned_result['conversation'] = cleaned_conversation

        else:
            # Handle other dict structures by cleaning all string values
            for key, value in result_data.items():
                if isinstance(value, str):
                    cleaned_result[key] = replace_track_ids_with_tokens(value, information)
                elif isinstance(value, list):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, dict):
                            cleaned_item = clean_result_data(item, information)
                            cleaned_list.append(cleaned_item)
                        elif isinstance(item, str):
                            cleaned_list.append(replace_track_ids_with_tokens(item, information))
                        else:
                            cleaned_list.append(item)
                    cleaned_result[key] = cleaned_list
                else:
                    cleaned_result[key] = value

        return cleaned_result

    elif isinstance(result_data, str):
        # If it's a raw string, try to parse as JSON first
        try:
            parsed_data = json.loads(result_data)
            return clean_result_data(parsed_data, information)
        except json.JSONDecodeError:
            # If not JSON, just clean the string directly
            return replace_track_ids_with_tokens(result_data, information)

    else:
        # Return as-is for other types
        return result_data


def process_example_with_cleaning(example_data, conversation=False):
    """
    Enhanced version of process_example that includes track ID cleaning.
    This function should replace the original process_example function in your code.
    """
    try:
        features = example_data['features']
        information = example_data['information']

        # Ensure we have evolutionary_stages and n_stages for all examples
        if 'evolutionary_stages' not in information:
            # Convert single stage to multi-stage format
            information['evolutionary_stages'] = [{
                'age': information.get('age', 0),
                'Teff': information.get('Teff', 0),
                'logg': information.get('logg', 0),
                'L': information.get('L', 0),
                'evolutionary_phase': information.get('evolutionary_phase', 'unknown')
            }]
            information['n_stages'] = 1
            information['stage_mapping'] = [{
                'stage_index': 0,
                'age': information.get('age', 0),
                'real_star_idx': information.get('real_star_idx', 0)
            }]

        # Ensure features are in the right format
        if information['n_stages'] == 1 and isinstance(features, np.ndarray) and len(features.shape) == 1:
            # Convert 1D features to 2D for consistency
            features = features.reshape(1, -1)

        # Get result from caption_star (your existing function)
        result = caption_star(features, information, conversation)

        # Clean the result to remove track ID references
        if result is not None:
            cleaned_result = clean_result_data(result, information)
        else:
            cleaned_result = None

        # Create return structure - now uniform for all examples
        return_data = {
            'track_id': information.get('track_id', 'unknown'),
            'star_id': information.get('star_id', 'unknown'),
            'n_stages': information.get('n_stages', 1),
            'evolutionary_stages': information.get('evolutionary_stages', []),
            'stage_mapping': information.get('stage_mapping', []),
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'result': cleaned_result,  # Use cleaned result
            'information': information
        }

        return return_data

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def save_results(results, filename):
    """
    Your existing function for saving captioning results to JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


class EvolutionaryDatasetCreator:
    """
    Creates a dataset matching evolutionary tracks with real stellar observations.
    Integrates with your existing prompt generation pipeline.
    """

    def __init__(self, real_stars_df, real_features,
                 feature_columns=['Teff', 'logg', 'L', 'Mstar'],
                 weights=None):
        """
        Initialize the dataset creator.

        Parameters:
        -----------
        real_stars_df : pd.DataFrame
            DataFrame with real star observations containing stellar parameters
        real_features : np.array
            Features array of shape (n_stars, hidden_dim) corresponding to real_stars_df
        feature_columns : list
            Column names in real_stars_df to use for matching
        weights : dict or None
            Weights for each parameter in matching (default: equal weights)
        """
        self.real_stars_df = real_stars_df.copy()
        self.real_features = real_features
        self.feature_columns = feature_columns

        # Set default weights if none provided
        if weights is None:
            self.weights = {col: 1.0 for col in feature_columns}
        else:
            self.weights = weights

        # Prepare real star data for matching
        self._prepare_real_star_data()

    def _prepare_real_star_data(self):
        """Prepare and normalize real star data for efficient matching."""
        print("Preparing real star data for matching...")

        # Extract relevant columns and handle missing values
        self.real_params = self.real_stars_df[self.feature_columns].copy()

        # Remove rows with any NaN values in the matching columns
        valid_mask = ~self.real_params.isnull().any(axis=1)
        self.real_params = self.real_params[valid_mask]
        self.real_features_clean = self.real_features[valid_mask]
        self.real_stars_clean = self.real_stars_df[valid_mask].reset_index(drop=True)

        print(f"Found {len(self.real_params)} valid real stars for matching")

        # Normalize parameters for fair comparison
        self.scaler = StandardScaler()
        self.real_params_normalized = self.scaler.fit_transform(self.real_params)

        # Apply weights
        weight_array = np.array([self.weights[col] for col in self.feature_columns])
        self.real_params_weighted = self.real_params_normalized * weight_array

    def find_closest_real_star(self, synthetic_params, max_deviations=None, min_deviations=None):
        """
        Find the closest real star to synthetic parameters using individual parameter thresholds.

        Parameters:
        -----------
        synthetic_params : dict
            Dictionary with stellar parameters {parameter: value}
        max_deviations : dict or None
            Maximum allowed absolute deviation for each parameter
            Example: {'Teff': 200, 'logg': 0.3, 'L': 0.5, 'Mstar': 0.2}
            If None, uses default values
        min_deviations : dict or None
            Minimum required absolute deviation for each parameter to avoid exact duplicates
            If None, uses small default values

        Returns:
        --------
        dict or None: Information about the matched real star, or None if no valid match
        """
        # Set default maximum deviations if not provided
        if max_deviations is None:
            max_deviations = {
                'Teff': 300,  # ±300 K
                'logg': 0.4,  # ±0.4 dex
                'L': 0.8,  # ±0.8 solar luminosities
                'Mstar': 0.3  # ±0.3 solar masses
            }

        # Set default minimum deviations if not provided
        if min_deviations is None:
            min_deviations = {
                'Teff': 10,  # At least 10 K difference
                'logg': 0.01,  # At least 0.01 dex difference
                'L': 0.01,  # At least 0.01 L_sun difference
                'Mstar': 0.01  # At least 0.01 M_sun difference
            }

        # Extract synthetic parameter values
        synth_values = np.array([synthetic_params[col] for col in self.feature_columns])

        # Create masks for each parameter constraint
        valid_mask = np.ones(len(self.real_params), dtype=bool)

        for i, param in enumerate(self.feature_columns):
            real_values = self.real_params.iloc[:, i].values
            synth_value = synth_values[i]

            # Calculate absolute differences for this parameter
            abs_diff = np.abs(real_values - synth_value)

            # Apply maximum deviation constraint
            max_dev = max_deviations.get(param, np.inf)
            max_constraint = abs_diff <= max_dev

            # Apply minimum deviation constraint
            min_dev = min_deviations.get(param, 0)
            min_constraint = abs_diff >= min_dev

            # Combine constraints for this parameter
            param_valid = max_constraint & min_constraint
            valid_mask = valid_mask & param_valid

        # Check if any stars meet all constraints
        if not np.any(valid_mask):
            return None

        # Among valid stars, find the one with minimum weighted distance
        valid_indices = np.where(valid_mask)[0]
        valid_real_params = self.real_params.iloc[valid_indices].values

        # Calculate weighted distances only for valid stars
        synth_tiled = np.tile(synth_values, (len(valid_indices), 1))
        differences = valid_real_params - synth_tiled

        # Apply weights to differences
        weight_array = np.array([self.weights[col] for col in self.feature_columns])
        weighted_differences = differences * weight_array

        # Calculate weighted Euclidean distances
        distances = np.sqrt(np.sum(weighted_differences ** 2, axis=1))

        # Find the closest valid match
        closest_valid_idx = np.argmin(distances)
        closest_idx = valid_indices[closest_valid_idx]

        # Calculate parameter-wise differences for the chosen star
        param_differences = {}
        for i, param in enumerate(self.feature_columns):
            real_val = self.real_params.iloc[closest_idx, i]
            synth_val = synth_values[i]
            param_differences[param] = {
                'real_value': real_val,
                'synthetic_value': synth_val,
                'absolute_difference': abs(real_val - synth_val),
                'relative_difference': abs(real_val - synth_val) / synth_val if synth_val != 0 else np.inf
            }

        return {
            'real_star_idx': closest_idx,
            'weighted_distance': distances[closest_valid_idx],
            'parameter_differences': param_differences,
            'real_params': self.real_params.iloc[closest_idx].to_dict(),
            'real_features': self.real_features_clean[closest_idx],
            'real_star_data': self.real_stars_clean.iloc[closest_idx].to_dict(),
            'n_candidates': len(valid_indices)  # How many stars met the criteria
        }

    def create_matched_dataset(self, n_evolutionary_tracks=100,
                               age_points=10, track_params=None,
                               max_deviations=None, min_deviations=None,
                               min_matches_per_track=3, create_evolutionary_prompts=True,
                               min_stages_for_evolution=2):
        """
        Create dataset by matching evolutionary tracks with real stars.
        Modified to ensure all outputs have the same structure.

        Parameters:
        -----------
        create_evolutionary_prompts : bool
            Whether to create evolutionary prompts for stars with multiple matches
        min_stages_for_evolution : int
            Minimum number of matched stages required to create evolutionary prompt

        Returns:
        --------
        list: List of examples ready for your process_example function
        """
        # Set default track parameters
        if track_params is None:
            track_params = {
                'n_stars': n_evolutionary_tracks,
                'age_points': age_points,
                'm_min': 0.5,
                'm_max': 2.0,
                'min_age': 0.8,
                'max_age': 10.0,
                'feh_range': (-0.5, 0.3),
                'grid_name': 'fastlaunch',
                'method': 'nearest'
            }

        print(f"Generating {n_evolutionary_tracks} evolutionary tracks...")

        # Generate evolutionary tracks
        tracks = generate_evolutionary_tracks(**track_params)
        tracks_df = tracks_to_dataframe(tracks)

        print(f"Generated tracks with {len(tracks_df)} total points")
        print(f"Searching for real star matches...")

        # Print matching criteria
        if max_deviations:
            print("Maximum allowed deviations:")
            for param, dev in max_deviations.items():
                print(f"  {param}: ±{dev}")

        # Store examples in unified format
        all_examples = []
        track_success_count = 0
        total_matches = 0

        # Process each evolutionary track
        unique_stars = tracks_df['star_id'].unique()

        for star_id in tqdm(unique_stars, desc="Processing evolutionary tracks"):
            star_track = tracks_df[tracks_df['star_id'] == star_id].copy()
            star_matches = []

            # Try to match each evolutionary stage of this star
            for stage_num, stage in star_track.iterrows():
                # Prepare parameters for matching
                synthetic_params = {
                    'Teff': stage['Teff'],
                    'logg': stage['logg'],
                    'L': stage['L'],  # Use your column naming
                    'Mstar': stage['mass']  # Use mass as Mstar estimate
                }

                # Find closest real star
                match = self.find_closest_real_star(
                    synthetic_params,
                    max_deviations=max_deviations,
                    min_deviations=min_deviations
                )

                if match is not None:
                    total_matches += 1

                    # Store match info with stage data
                    match_info = {
                        'stage_data': stage,
                        'match_result': match,
                        'age': stage['age'],
                        'Teff': stage['Teff'],
                        'logg': stage['logg'],
                        'L': stage['L'],
                        'mass': stage['mass'],
                        'feh': stage['feh'],
                        'evolutionary_phase': 'main_sequence' if giant_cond(stage['Teff'], stage['logg']) else 'giant'
                    }
                    star_matches.append(match_info)

            # Process this star's matches - now accept single matches too
            if len(star_matches) >= 1:  # Changed from min_matches_per_track to 1
                track_success_count += 1

                # Sort matches by age
                star_matches.sort(key=lambda x: x['age'])

                # Always create examples in the unified format (multi-stage structure)
                if len(star_matches) == 1:
                    # Single stage - create as 1-stage multi-stage example
                    match_info = star_matches[0]
                    stage = match_info['stage_data']
                    match = match_info['match_result']

                    # Create evolutionary stages list with single stage
                    evolutionary_stages = [{
                        'age': stage['age'],
                        'Teff': stage['Teff'],
                        'logg': stage['logg'],
                        'L': stage['L'],
                        'evolutionary_phase': match_info['evolutionary_phase']
                    }]

                    information = {
                        'track_id': int(star_id),
                        'star_id': f"track_{star_id}_age_{stage['age']:.2f}",
                        'initial_mass': stage['mass'],
                        'metallicity': stage['feh'],
                        'evolutionary_stages': evolutionary_stages,
                        'n_stages': 1,
                        'age_span': 0.0,
                        'stage_mapping': [{
                            'stage_index': 0,
                            'age': stage['age'],
                            'real_star_idx': int(match['real_star_idx'])
                        }],
                        'match_distance': match['weighted_distance'],
                        'parameter_differences': match['parameter_differences'],
                        'real_star_params': match['real_params'],
                        'n_candidates': match['n_candidates']
                    }

                    example = {
                        'features': match['real_features'].reshape(1, -1),  # Ensure 2D
                        'information': information
                    }
                    all_examples.append(example)

                elif len(star_matches) >= min_matches_per_track:
                    # Multiple stages - create multi-stage example (only if meets minimum threshold)
                    evolutionary_stages = []
                    all_features = []
                    stage_mapping = []

                    for i, match_info in enumerate(star_matches):
                        stage_info = {
                            'age': match_info['age'],
                            'Teff': match_info['Teff'],
                            'logg': match_info['logg'],
                            'L': match_info['L'],
                            'evolutionary_phase': match_info['evolutionary_phase']
                        }
                        evolutionary_stages.append(stage_info)
                        all_features.append(match_info['match_result']['real_features'])
                        stage_mapping.append({
                            'stage_index': i,
                            'age': match_info['age'],
                            'real_star_idx': int(match_info['match_result']['real_star_idx'])
                        })

                    # Create evolutionary information
                    evolutionary_info = {
                        'track_id': int(star_id),
                        'star_id': f"track_{star_id}_evolutionary",
                        'initial_mass': star_matches[0]['mass'],
                        'metallicity': star_matches[0]['feh'],
                        'evolutionary_stages': evolutionary_stages,
                        'n_stages': len(evolutionary_stages),
                        'age_span': star_matches[-1]['age'] - star_matches[0]['age'],
                        'stage_mapping': stage_mapping
                    }

                    evolutionary_example = {
                        'features': np.array(all_features),  # Multiple features
                        'information': evolutionary_info
                    }
                    all_examples.append(evolutionary_example)

        print(f"Successfully matched {track_success_count}/{len(unique_stars)} tracks")
        print(f"Total examples created: {len(all_examples)}")
        print(f"Total successful matches: {total_matches}")

        # Print statistics
        single_stage_count = sum(1 for ex in all_examples if ex['information']['n_stages'] == 1)
        multi_stage_count = len(all_examples) - single_stage_count

        print(f"Single-stage examples (n_stages=1): {single_stage_count}")
        print(f"Multi-stage examples (n_stages>1): {multi_stage_count}")

        # Print matching statistics
        if all_examples:
            print("\nMatching quality statistics:")
            all_stage_mappings = []
            for ex in all_examples:
                all_stage_mappings.extend(ex['information'].get('stage_mapping', []))

            print(f"Total individual stage matches: {len(all_stage_mappings)}")

        return all_examples


def main(real_stars_df, real_features, n_tracks=100, age_points=10,
         output_file='stellar_evolution_dataset.json', conversation=False,
         use_multiprocessing=True, n_processes=4, max_deviations=None, min_deviations=None,
         create_evolutionary_prompts=True, min_stages_for_evolution=2,
         include_single_stages=True):
    """
    Main function using your existing pipeline structure with unified output format.

    Parameters:
    -----------
    create_evolutionary_prompts : bool
        Whether to create multi-stage evolutionary prompts (legacy parameter, now always creates unified format)
    min_stages_for_evolution : int
        Minimum number of matched stages to create evolutionary prompt
    include_single_stages : bool
        Whether to include single-stage matches (n_stages=1) in the dataset
    """

    # Create the dataset creator
    creator = EvolutionaryDatasetCreator(
        real_stars_df=real_stars_df,
        real_features=real_features,
        feature_columns=['Teff', 'logg', 'L', 'Mstar'],
        weights={'Teff': 1.0, 'logg': 1.5, 'L': 1.0, 'Mstar': 2.0}
    )

    # Create matched examples
    print("Creating matched evolutionary dataset...")
    examples = creator.create_matched_dataset(
        n_evolutionary_tracks=n_tracks,
        age_points=age_points,
        max_deviations=max_deviations,
        min_deviations=min_deviations,
        min_matches_per_track=min_stages_for_evolution if not include_single_stages else 1,
        create_evolutionary_prompts=create_evolutionary_prompts,
        min_stages_for_evolution=min_stages_for_evolution
    )

    if len(examples) == 0:
        print("No valid examples created. Try relaxing the matching criteria.")
        return

    # All examples now have unified structure
    single_stage = [ex for ex in examples if ex['information'].get('n_stages') == 1]
    multi_stage = [ex for ex in examples if ex['information'].get('n_stages', 1) > 1]

    print(f"Processing {len(examples)} examples with Gemini...")
    print(f"  - Single-stage examples (n_stages=1): {len(single_stage)}")
    print(f"  - Multi-stage examples (n_stages>1): {len(multi_stage)}")

    # Process examples using your existing pipeline
    if use_multiprocessing and n_processes > 1:
        # Use multiprocessing as in your original code
        process_func = partial(process_example_with_cleaning, conversation=conversation)

        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, examples),
                total=len(examples),
                desc="Processing with Gemini"
            ))
    else:
        # Process sequentially
        results = []
        for example in tqdm(examples, desc="Processing with Gemini"):
            result = process_example_with_cleaning(example, conversation=conversation)
            results.append(result)

    # Filter out failed results
    successful_results = [r for r in results if r is not None and r['result'] is not None]

    # Separate successful results by number of stages
    successful_single = [r for r in successful_results if r.get('n_stages') == 1]
    successful_multi = [r for r in successful_results if r.get('n_stages', 1) > 1]

    print(f"Successfully processed {len(successful_results)}/{len(examples)} examples")
    print(f"  - Single-stage success (n_stages=1): {len(successful_single)}/{len(single_stage) if single_stage else 0}")
    print(f"  - Multi-stage success (n_stages>1): {len(successful_multi)}/{len(multi_stage) if multi_stage else 0}")

    # Save using your existing function
    save_results(successful_results, output_file)

    # Create summary with detailed statistics
    summary = {
        'metadata': {
            'total_examples': len(examples),
            'successful_results': len(successful_results),
            'single_stage_examples': len(single_stage),
            'multi_stage_examples': len(multi_stage),
            'single_stage_success': len(successful_single),
            'multi_stage_success': len(successful_multi),
            'conversation_mode': conversation,
            'feature_dim': real_features.shape[1] if len(real_features.shape) > 1 else len(real_features[0]),
            'max_deviations': max_deviations,
            'min_deviations': min_deviations,
            'unified_output_format': True  # Flag indicating all outputs have same structure
        },
        'results': successful_results
    }

    # Save complete dataset
    final_output = output_file.replace('.json', '_complete.json')
    save_results(summary, final_output)

    print(f"Dataset creation complete!")
    print(f"- Results saved to: {output_file}")
    print(f"- Complete dataset saved to: {final_output}")
    print(f"- Overall success rate: {len(successful_results) / len(examples) * 100:.1f}%")
    print(
        f"- Single-stage success rate: {len(successful_single) / len(single_stage) * 100:.1f}%" if single_stage else "- No single-stage examples")
    print(
        f"- Multi-stage success rate: {len(successful_multi) / len(multi_stage) * 100:.1f}%" if multi_stage else "- No multi-stage examples")
    print("- All examples now use unified output structure with n_stages field")

    return summary



# Example usage
if __name__ == "__main__":
    # Example with real data pipeline (your exact code)
    print("Creating stellar evolution dataset with unified output format...")

    # Load your real data
    latent_features = np.load('/data/TalkingLatents/logs/2025-07-29/features.npy')
    metadata_df = pd.read_csv('/data/TalkingLatents/logs/2025-07-29/info.csv')
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]
    berger_lamost = pd.read_csv('/data/TalkingLatents/tables/lamost_dr8_with_berger_labels.csv')
    berger_lamost = berger_lamost.loc[:, ~berger_lamost.columns.str.contains('^Unnamed')]

    real_stars_df, real_features = merge_dataframes_and_filter_array(
        berger_lamost, metadata_df, 'obsid', 'obsid', latent_features
    )
    real_stars_df.rename(columns={'Lstar': 'L'}, inplace=True)

    # Run the complete pipeline with unified output format
    result = main(
        real_stars_df=real_stars_df,
        real_features=real_features,
        n_tracks=10000,
        age_points=4,
        output_file='/data/TalkingLatents/data/dataset/stellar_evolution_results.json',
        conversation=False,  # Set to True for conversation mode
        use_multiprocessing=True,  # Set to True for parallel processing
        n_processes=16,
        max_deviations={
            'Teff': 100,  # ±100 K temperature difference
            'logg': 0.2,  # ±0.2 dex surface gravity difference
            'L': 1,  # ±1 solar luminosity difference
            'Mstar': 0.2  # ±0.2 solar mass difference
        },
        min_deviations={
            'Teff': 0,
            'logg': 0,
            'L': 0,
            'Mstar': 0
        },
        include_single_stages=True  # Set to True to include single-stage matches
    )

    print("\nDataset composition:")
    print(f"Total examples: {result['metadata']['total_examples']}")
    print(f"Single-stage examples (n_stages=1): {result['metadata']['single_stage_examples']}")
    print(f"Multi-stage examples (n_stages>1): {result['metadata']['multi_stage_examples']}")

    print("\nUnified output structure:")
    print("All examples now have the same dictionary structure with:")
    print("- n_stages: number of evolutionary stages")
    print("- evolutionary_stages: list of stage information")
    print("- stage_mapping: mapping of stages to real star indices")
    print("- features: array of features (shape: [n_stages, feature_dim])")

    print("\nSample results:")
    if result['results']:
        # Show single-stage example (n_stages=1)
        single_examples = [r for r in result['results'] if r.get('n_stages') == 1]
        if single_examples:
            sample = single_examples[0]
            print(f"\nSingle-stage example (Track {sample['track_id']}, n_stages={sample['n_stages']}):")
            print(f"Features shape: {np.array(sample['features']).shape}")
            print(f"Evolutionary stages: {len(sample['evolutionary_stages'])}")
            print(f"Result preview: {str(sample['result'])[:150]}...")

        # Show multi-stage example (n_stages>1)
        multi_examples = [r for r in result['results'] if r.get('n_stages', 1) > 1]
        if multi_examples:
            sample = multi_examples[0]
            print(f"\nMulti-stage example (Track {sample['track_id']}, n_stages={sample['n_stages']}):")
            print(f"Features shape: {np.array(sample['features']).shape}")
            print(f"Evolutionary stages: {len(sample['evolutionary_stages'])}")
            print(f"Result preview: {str(sample['result'])[:150]}...")

            # Show stage mapping
            print("Stage mapping:")
            for stage in sample['stage_mapping']:
                print(f"  Stage {stage['stage_index']}: Age {stage['age']:.2f} Gyr")

    print("\nKey improvements:")
    print("1. ✅ Unified output structure: All examples have the same dictionary format")
    print("2. ✅ Length limits: Added word count limits to Gemini prompts (400 words total, 300 for answers)")
    print("3. ✅ Single-stage as multi-stage: Single stages are now treated as 1-stage multi-stage examples")
    print("4. ✅ Consistent features format: All features are 2D arrays [n_stages, feature_dim]")