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
import time

warnings.filterwarnings('ignore')


# Import your existing functions (assuming they're available)

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
        if isinstance(value, (int, float)) and not pd.isna(value):
            # Format temperatures
            if key in ['Teff', 'temperature']:
                return f"{float(value):.0f}"
            # Format surface gravity
            elif key in ['logg', 'surface_gravity']:
                return f"{float(value):.2f}"
            # Format metallicity
            elif key in ['feh', 'metallicity', 'FeH']:
                return f"{float(value):.2f}"
            # Format masses
            elif key in ['mass', 'Mstar', 'initial_mass']:
                return f"{float(value):.2f}"
            # Format luminosity
            elif key in ['Lstar', 'luminosity', 'L']:
                return f"{float(value):.3f}"
            # Format age
            elif key in ['age']:
                return f"{float(value):.2f}"
            # Default number formatting
            else:
                return f"{float(value):.4f}" if isinstance(value, float) else f"{int(value)}"

        # String values - ensure no extra spaces
        if isinstance(value, str):
            return value.strip()

        return str(value)

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
            'Main-Sequence', 'Giant', 'main_sequence', 'giant', 'evolutionary_phase'
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

    formatted_prompt += f"""

Based on this information, create a diverse, detailed scientific description.
Create a unique description about this star's evolution that:
- Asks about evolutionary processes, stellar physics, or temporal changes
- Could focus on any aspect: mass loss, nuclear burning, structural changes, HR diagram evolution, etc.
- Should sound like a real research question about stellar evolution

IMPORTANT: Keep your response concise and under 100 words total.
"""

    return formatted_prompt


def caption_star(information):
    """
    Generate plain text stellar description using Gemini
    """
    global client
    if client is None:
        print("Warning: Gemini client not available")
        return None

    prompt = create_star_prompt(information)

    # Generate response from Gemini in regular text mode
    try:
        response = client.models.generate_content(
            contents=[prompt],
            model="gemini-2.0-flash"
        )
        response_text = response.text.strip()

        # Clean up any unwanted prefixes or formatting
        prefixes_to_remove = [
            "designation: undetermined.",
            "designation:",
            "star description:",
            "description:",
            "stellar description:",
            "**star description:**",
            "**description:**",
            "analysis reveals",
            "analysis:"
        ]

        response_lower = response_text.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
                break

        # Remove any markdown formatting or asterisks at the beginning
        response_text = re.sub(r'^[\*\#\-\s]+', '', response_text)

        # Remove newlines and extra whitespace, replace with single spaces
        response_text = re.sub(r'\s+', ' ', response_text)

        # Fix spacing around decimal points and equals signs - more aggressive approach
        response_text = re.sub(r'\s*=\s*', ' = ', response_text)  # Normalize spacing around equals

        # Fix decimal points with spaces - multiple patterns to catch all cases
        response_text = re.sub(r'(\d)\s+\.\s*(\d)', r'\1.\2', response_text)  # "4 . 63" -> "4.63"
        response_text = re.sub(r'(\d)\s*\.\s+(\d)', r'\1.\2', response_text)  # "4. 63" -> "4.63"
        response_text = re.sub(r'(\d)\s+\.(\d)', r'\1.\2', response_text)  # "4 .63" -> "4.63"

        # Fix negative signs with spaces
        response_text = re.sub(r'-\s+(\d)', r'-\1', response_text)  # "- 0.39" -> "-0.39"
        response_text = re.sub(r'=\s+-\s+(\d)', r'= -\1', response_text)  # "= - 0.39" -> "= -0.39"

        # Fix specific scientific notation patterns
        response_text = re.sub(r'log\s*g\s*=\s*(\d+)\s*\.\s*(\d+)', r'log g = \1.\2', response_text)
        response_text = re.sub(r'\[Fe/H\]\s*=\s*-?\s*(\d+)\s*\.\s*(\d+)', r'[Fe/H] = -\1.\2', response_text)
        response_text = re.sub(r'\[Fe/H\]\s*=\s*(\d+)\s*\.\s*(\d+)', r'[Fe/H] = \1.\2',
                               response_text)  # positive values

        # General cleanup for any remaining decimal spacing issues
        response_text = re.sub(r'(\d)\s+\.(\d)', r'\1.\2', response_text)
        response_text = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', response_text)

        # Fix temperature values too
        response_text = re.sub(r'(\d+)\s+K\b', r'\1 K', response_text)  # Ensure single space before K

        # Remove questions at the end (sentences ending with ?)
        sentences = response_text.split('.')
        descriptive_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith('?'):
                descriptive_sentences.append(sentence)

        # Rejoin sentences
        if descriptive_sentences:
            response_text = '. '.join(descriptive_sentences)
            if not response_text.endswith('.'):
                response_text += '.'

        return response_text.strip()

    except Exception as e:
        print(f"Error generating content: {e}")
        return None


def process_single_star(row_data, obsid_column='obsid', max_retries=3, delay_between_requests=0.1):
    """
    Process a single star row and generate its description

    Args:
        row_data (tuple): (index, row_series) from dataframe iterrows()
        obsid_column (str): Name of the observation ID column
        max_retries (int): Maximum number of retries for failed requests
        delay_between_requests (float): Delay between API requests to avoid rate limits

    Returns:
        dict: Star data with description, or None if failed
    """
    try:
        idx, row = row_data

        # Convert row to dictionary
        star_info = row.to_dict()

        # Add derived information
        if 'Teff' in star_info and 'logg' in star_info:
            star_info['is_giant'] = not giant_cond(star_info['Teff'], star_info['logg'])

        # Generate description with retries
        description = None
        for attempt in range(max_retries):
            try:
                time.sleep(delay_between_requests)  # Rate limiting
                description = caption_star(star_info)
                if description:
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {obsid_column} {star_info.get(obsid_column, idx)}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_between_requests * (attempt + 1))  # Exponential backoff

        # Prepare output data
        output_data = {
            'index': idx,
            obsid_column: star_info.get(obsid_column, None),
            'description': description,
            'stellar_data': star_info,
            'processing_timestamp': time.time()
        }

        return output_data

    except Exception as e:
        print(f"Error processing star at index {idx}: {e}")
        return None


def process_star_worker(args):
    """
    Worker function for multiprocessing - initializes client and processes star

    Args:
        args (tuple): (row_data, obsid_column, max_retries, delay_between_requests)

    Returns:
        dict: Processed star data or None
    """
    global client

    # Initialize client for this worker process
    if client is None:
        client = init_gemini_client()

    row_data, obsid_column, max_retries, delay_between_requests = args
    return process_single_star(row_data, obsid_column, max_retries, delay_between_requests)


def generate_stellar_descriptions(df,
                                  output_file='stellar_descriptions.json',
                                  obsid_column='obsid',
                                  n_processes=4,
                                  max_retries=3,
                                  delay_between_requests=0.1,
                                  save_intermediate=True,
                                  intermediate_interval=100):
    """
    Generate descriptions for all stars in the dataframe and save as JSON

    Args:
        df (pd.DataFrame): Dataframe containing stellar information
        output_file (str): Output JSON filename
        obsid_column (str): Name of the observation ID column
        n_processes (int): Number of processes for multiprocessing
        max_retries (int): Maximum retries for failed API requests
        delay_between_requests (float): Delay between API requests
        save_intermediate (bool): Whether to save intermediate results
        intermediate_interval (int): Save intermediate results every N processed stars

    Returns:
        list: List of processed star data dictionaries
    """
    global client

    print(f"Processing {len(df)} stars with {n_processes} processes...")
    print(f"Output will be saved to: {output_file}")

    # Initialize the global client (for the main process)
    client = init_gemini_client()

    # Prepare arguments for multiprocessing
    args_list = [
        (row_data, obsid_column, max_retries, delay_between_requests)
        for row_data in df.iterrows()
    ]

    # Process with multiprocessing
    results = []
    failed_count = 0

    if n_processes > 1:
        # Use multiprocessing
        with Pool(processes=n_processes) as pool:
            # Use imap for progress tracking
            with tqdm(total=len(args_list), desc="Processing stars") as pbar:
                for i, result in enumerate(pool.imap(process_star_worker, args_list)):
                    if result is not None:
                        results.append(result)
                    else:
                        failed_count += 1

                    pbar.update(1)

                    # Save intermediate results
                    if save_intermediate and (i + 1) % intermediate_interval == 0:
                        intermediate_file = f"{output_file.rsplit('.', 1)[0]}_intermediate_{i + 1}.json"
                        with open(intermediate_file, 'w') as f:
                            json.dump(results, f, indent=2, default=str)
                        print(f"\nSaved intermediate results to {intermediate_file}")
    else:
        # Single process with progress bar
        for i, args in enumerate(tqdm(args_list, desc="Processing stars")):
            result = process_star_worker(args)
            if result is not None:
                results.append(result)
            else:
                failed_count += 1

            # Save intermediate results
            if save_intermediate and (i + 1) % intermediate_interval == 0:
                intermediate_file = f"{output_file.rsplit('.', 1)[0]}_intermediate_{i + 1}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nSaved intermediate results to {intermediate_file}")

    # Save final results
    print(f"\nProcessing complete! Successfully processed: {len(results)}, Failed: {failed_count}")

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_file}")

    # Print some statistics
    successful_descriptions = sum(1 for r in results if r.get('description') is not None)
    print(f"Stars with successful descriptions: {successful_descriptions}/{len(results)}")

    return results


def load_and_validate_dataframe(df, required_columns=None, obsid_column='obsid'):
    """
    Validate and prepare dataframe for processing

    Args:
        df (pd.DataFrame): Input dataframe
        required_columns (list): List of required columns
        obsid_column (str): Name of observation ID column

    Returns:
        pd.DataFrame: Validated dataframe
    """
    if required_columns is None:
        required_columns = [obsid_column]

    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing obsid
    if obsid_column in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=[obsid_column])
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} rows with missing {obsid_column}")

    print(f"Dataframe validated. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


# Initialize global client
client = None

if __name__ == "__main__":
    # Example with synthetic data for demonstration
    np.random.seed(42)
    n_stars = 100

    df = pd.read_csv('/data/TalkingLatents/logs/2025-07-29/info.csv')
    df['Teff'] *= 5778
    # df = df.iloc[:n_stars]

    generate_stellar_descriptions(df,
     output_file='/data/TalkingLatents/data/dataset/stellar_descriptions.json',
     n_processes=16)

    # Uncomment to run with synthetic data
    # df = load_and_validate_dataframe(df, obsid_column='obsid')
    # results = generate_stellar_descriptions(df, n_processes=2, intermediate_interval=10)