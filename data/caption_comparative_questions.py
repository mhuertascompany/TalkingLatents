import argparse
import json
import math
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
os.system('pip install google-genai kiauhoku')
from google import genai

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from data.stellar_evolution import interpolate_stellar_parameters


ROOT_DIR = Path(__file__).resolve().parents[2]
GOOGLE_KEY_PATH = Path("google_api.txt")

# Required columns for generating correct answers (not shown in questions)
REQUIRED_DERIVED_PARAMS = ("Teff", "logg", "FeH")
DEFAULT_MASS_GRID = np.linspace(0.7, 3.0, 50)
DEFAULT_AGE_GRID = np.linspace(0.1, 13.0, 50)
KELVIN_SCALE = 5778.0
TEMPERATURE_THRESHOLD = 150.0
AGE_THRESHOLD = 0.6
METALLICITY_THRESHOLD = 0.1

OPTION_SETS = {
    "temperature": [
        ("A", "Star A is hotter than Star B."),
        ("B", "Star B is hotter than Star A."),
        ("C", "Both stars have comparable temperatures."),
        ("D", "There is insufficient information to compare their temperatures."),
    ],
    "age": [
        ("A", "Star A is older than Star B."),
        ("B", "Star B is older than Star A."),
        ("C", "They are approximately the same age."),
        ("D", "Their ages cannot be inferred from the data provided."),
    ],
    "metallicity": [
        ("A", "Star A is more metal-rich than Star B."),
        ("B", "Star B is more metal-rich than Star A."),
        ("C", "They have effectively the same metallicity."),
        ("D", "The metallicities are not known with sufficient accuracy."),
    ],
    "stage": [
        ("A", "Star A is the more evolved giant while Star B remains on the main sequence."),
        ("B", "Star B is the more evolved giant while Star A remains on the main sequence."),
        ("C", "Both stars are on the same evolutionary stage."),
        ("D", "The evolutionary stages cannot be determined from the data."),
    ],
}

COMPARISON_DESCRIPTIONS = {
    "temperature": "which star has the higher effective temperature",
    "age": "which star is older",
    "metallicity": "which star has the higher metallicity",
    "stage": "which star is more evolved",
}


def giant_cond(teff: float, logg: float) -> bool:
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4.0
    else:
        thresh = 5.2 - (2.8e-4 * teff)
    return logg >= thresh


def parse_gemini_json_response(response_text: str) -> Optional[Dict[str, object]]:
    text = response_text.strip()
    patterns = [
        r"^```json\s*(.*?)\s*```$",
        r"^```\s*(.*?)\s*```$",
        r"^`(.*?)`$",
    ]

    for pattern in patterns:
        match = __import__("re").search(pattern, text, __import__("re").DOTALL)
        if match:
            text = match.group(1).strip()
            break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def init_gemini_client() -> Optional[genai.Client]:
    if not GOOGLE_KEY_PATH.exists():
        print("Warning: google_api.txt not found; Gemini client disabled.")
        return None
    api_key = GOOGLE_KEY_PATH.read_text().strip()
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        print(f"Warning: failed to initialise Gemini client: {exc}")
        return None


class EvolutionInference:
    def __init__(
        self,
        mass_grid: Iterable[float] = DEFAULT_MASS_GRID,
        age_grid: Iterable[float] = DEFAULT_AGE_GRID,
        feh_grid: Optional[Iterable[float]] = None,
        grid_name: str = "mist",
        method: str = "nearest",
    ) -> None:
        self.mass_grid = np.array(list(mass_grid), dtype=np.float32)
        self.age_grid = np.array(list(age_grid), dtype=np.float32)
        
        # Default metallicity grid from -1.5 to +0.8 in 0.1 steps
        if feh_grid is None:
            self.feh_grid = np.arange(-1.5, 0.9, 0.1, dtype=np.float32)
        else:
            self.feh_grid = np.array(list(feh_grid), dtype=np.float32)
            
        self.grid_name = grid_name
        self.method = method
        
        # Pre-computed grids - will be populated by build_dense_grid()
        self.grid_points = None  # Shape: (N, 3) for (mass, feh, age)
        self.grid_teff = None
        self.grid_logg = None
        self.grid_luminosity = None
        self.grid_radius = None
        
        print(f"Initializing EvolutionInference with {len(self.mass_grid)} masses x "
              f"{len(self.feh_grid)} metallicities x {len(self.age_grid)} ages = "
              f"{len(self.mass_grid) * len(self.feh_grid) * len(self.age_grid)} grid points")

    def build_dense_grid(self) -> None:
        """Build a single dense 3D grid covering all (mass, metallicity, age) combinations"""
        start_time = time.time()
        
        # Create 3D meshgrid
        masses, fehs, ages = np.meshgrid(self.mass_grid, self.feh_grid, self.age_grid, indexing="ij")
        
        # Flatten to 1D arrays
        flat_mass = masses.ravel()
        flat_feh = fehs.ravel()
        flat_age = ages.ravel()
        flat_alpha = np.zeros_like(flat_mass, dtype=np.float32)
        
        # Single call to interpolate_stellar_parameters with progress indication
        with tqdm(total=1, desc="Building stellar evolution grid", unit="grid") as pbar:
            pbar.set_postfix({"Points": f"{len(flat_mass):,}"})
            
            preds = interpolate_stellar_parameters(
                flat_mass,
                flat_feh,
                flat_alpha,
                flat_age,
                grid_name=self.grid_name,
                method=self.method,
            )
            pbar.update(1)
        
        # Store the results
        self.grid_points = np.column_stack([flat_mass, flat_feh, flat_age])
        self.grid_teff = preds["Teff"]
        self.grid_logg = preds["logg"]
        self.grid_luminosity = preds["L"]
        self.grid_radius = preds["R"]
        
        elapsed = time.time() - start_time
        print(f"Grid built in {elapsed:.2f}s with {len(self.grid_points):,} points")

    def infer(self, teff: float, logg: float, feh: float) -> Dict[str, float]:
        if self.grid_points is None:
            raise RuntimeError("Must call build_dense_grid() before inference")

        # Find best match using temperature and surface gravity
        temp_term = ((self.grid_teff - teff) / 300.0) ** 2
        logg_term = ((self.grid_logg - logg) / 0.2) ** 2
        best_idx = int(np.argmin(temp_term + logg_term))
        
        return {
            "mass_solar": float(self.grid_points[best_idx, 0]),
            "age_gyr": float(self.grid_points[best_idx, 2]),
            "teff_model": float(self.grid_teff[best_idx]),
            "logg_model": float(self.grid_logg[best_idx]),
            "luminosity": float(self.grid_luminosity[best_idx]),
            "radius": float(self.grid_radius[best_idx]),
        }


def _as_kelvin(value: float) -> float:
    return value * KELVIN_SCALE if value < 1000 else value


def _clean(value: float, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return float(value)


def summarise_star(row: pd.Series, evo: EvolutionInference) -> Dict[str, Optional[float]]:
    raw_teff = _clean(row.get("Teff"))
    teff = _as_kelvin(raw_teff) if not math.isnan(raw_teff) else float("nan")
    logg = _clean(row.get("logg"))
    feh = _clean(row.get("FeH"), 0.0)

    inference = evo.infer(teff, logg, feh)
    stage = "dwarf" if giant_cond(teff, logg) else "giant"

    # Calculate parameter errors (observed - model)
    teff_error = teff - inference["teff_model"] if not math.isnan(teff) else None
    logg_error = logg - inference["logg_model"] if not math.isnan(logg) else None

    return {
        "obsid": row.get("obsid"),
        "teff_k": round(teff, 1) if not math.isnan(teff) else None,
        "logg": round(logg, 3) if not math.isnan(logg) else None,
        "feh": round(feh, 3),
        "estimated_age_gyr": round(inference["age_gyr"], 2),
        "estimated_mass_msun": round(inference["mass_solar"], 3),
        "estimated_radius_rsun": round(inference["radius"], 3),
        "estimated_luminosity_lsun": round(inference["luminosity"], 3),
        "stage": stage,
        # Model values for comparison
        "teff_model_k": round(inference["teff_model"], 1),
        "logg_model": round(inference["logg_model"], 3),
        # Parameter errors (observed - model)
        "teff_error_k": round(teff_error, 1) if teff_error is not None else None,
        "logg_error": round(logg_error, 3) if logg_error is not None else None,
        # Absolute errors for uncertainty estimation
        "teff_abs_error_k": round(abs(teff_error), 1) if teff_error is not None else None,
        "logg_abs_error": round(abs(logg_error), 3) if logg_error is not None else None,
    }


def choose_comparison(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> str:
    if a["stage"] != b["stage"]:
        return "stage"

    age_diff = abs(a["estimated_age_gyr"] - b["estimated_age_gyr"])
    if age_diff >= AGE_THRESHOLD:
        return "age"

    temp_diff = abs(a["teff_k"] - b["teff_k"])
    if temp_diff >= TEMPERATURE_THRESHOLD:
        return "temperature"

    metal_diff = abs(a["feh"] - b["feh"])
    if metal_diff >= METALLICITY_THRESHOLD:
        return "metallicity"

    return "temperature"


def compute_correct_label(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]], comparison: str) -> Tuple[str, str]:
    if comparison == "temperature":
        diff = a["teff_k"] - b["teff_k"]
        if abs(diff) < TEMPERATURE_THRESHOLD:
            return "C", "Temperatures are close enough to be considered comparable."
        if diff > 0:
            return "A", f"Star A is hotter by {diff:.0f} K."
        return "B", f"Star B is hotter by {abs(diff):.0f} K."

    if comparison == "age":
        diff = a["estimated_age_gyr"] - b["estimated_age_gyr"]
        if abs(diff) < AGE_THRESHOLD:
            return "C", "Ages are effectively indistinguishable at this precision."
        if diff > 0:
            return "A", f"Star A is older by {diff:.2f} Gyr."
        return "B", f"Star B is older by {abs(diff):.2f} Gyr."

    if comparison == "metallicity":
        diff = a["feh"] - b["feh"]
        if abs(diff) < METALLICITY_THRESHOLD:
            return "C", "Metallicities differ by less than the chosen threshold."
        if diff > 0:
            return "A", f"Star A has higher [Fe/H] by {diff:.2f} dex."
        return "B", f"Star B has higher [Fe/H] by {abs(diff):.2f} dex."

    # stage comparison
    if a["stage"] == b["stage"]:
        return "C", "Both stars occupy the same evolutionary category."
    if a["stage"] == "giant":
        return "A", "Star A shows giant-like low gravity compared to Star B."
    return "B", "Star B shows giant-like low gravity compared to Star A."


# format_star_block function removed - no longer needed since questions use placeholders


def create_comparison_prompt(
    comparison: str,
    correct_label: str,
) -> str:
    """Create a prompt for Gemini to generate diverse questions with placeholders"""
    option_lines = [f"{label}. {text}" for label, text in OPTION_SETS[comparison]]
    description = COMPARISON_DESCRIPTIONS[comparison]

    prompt = f"""
You are an expert astrophysicist asked to craft a multiple-choice question comparing two stars.
Create a diverse and engaging question that asks {description}.

CRITICAL REQUIREMENTS:
1. The question MUST contain exactly one "STAR_A" placeholder and exactly one "STAR_B" placeholder.
2. Do NOT include any stellar data, parameters, measurements, or observational information in the question text.
3. The placeholders STAR_A and STAR_B will be replaced later with actual stellar data.
4. Focus on {description} but phrase it in a diverse, scientifically interesting way.
5. Keep the question under 30 words.
6. The question should be suitable for an astrophysics expert to answer based on stellar data.

EXAMPLES of good placeholder questions:
- "Comparing STAR_A and STAR_B, which exhibits characteristics typical of a hotter stellar object?"
- "Between STAR_A and STAR_B, which shows properties consistent with greater stellar evolution?"
- "Which star, STAR_A or STAR_B, displays features indicative of higher metallicity?"

Provide EXACTLY the four answer options listed below. Do not alter their wording or labels.
{os.linesep.join('   ' + line for line in option_lines)}

Return ONLY a JSON object with this structure:
{{
  "Question": "...",
  "Options": [{{"label": "A", "text": "..."}}, {{"label": "B", "text": "..."}}, {{"label": "C", "text": "..."}}, {{"label": "D", "text": "..."}}],
  "Answer": "{correct_label}",
  "Rationale": "Brief explanation based on stellar physics principles"
}}
Do not wrap the JSON in markdown fences.
"""
    return prompt.strip()


def generate_comparative_question(
    client: Optional[genai.Client],
    star_a: Dict[str, Optional[float]],
    star_b: Dict[str, Optional[float]],
    raw_data_a: pd.Series,
    raw_data_b: pd.Series,
    comparison: str,
    correct_label: str,
    correct_fact: str,
    max_retries: int = 3,
    delay: float = 0.25,
) -> Optional[Dict[str, object]]:
    if client is None:
        return None

    prompt = create_comparison_prompt(comparison, correct_label)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                contents=[prompt],
                model="gemini-2.0-flash",
                config={"response_mime_type": "application/json"},
            )
            parsed = parse_gemini_json_response(response.text)
            if parsed and "STAR_A" in parsed.get("Question", "") and "STAR_B" in parsed.get("Question", ""):
                return parsed
            else:
                print(f"Generated question missing required placeholders: {parsed.get('Question', '') if parsed else 'No response'}")
        except Exception as exc:
            print(f"Gemini request failed (attempt {attempt + 1}): {exc}")
        time.sleep(delay * (attempt + 1))
    return None


def sample_pairs(df: pd.DataFrame, num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    indices = df.index.to_list()
    selected: List[Tuple[int, int]] = []
    seen = set()
    while len(selected) < num_pairs and len(seen) < len(indices) * (len(indices) - 1) // 2:
        a, b = rng.choice(indices, size=2, replace=False)
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        selected.append(key)
    return selected


def process_pair(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    evo: EvolutionInference,
    client: Optional[genai.Client],
    max_retries: int,
    delay: float,
) -> Optional[Dict[str, object]]:
    star_a = summarise_star(df.loc[idx_a], evo)
    star_b = summarise_star(df.loc[idx_b], evo)
    comparison = choose_comparison(star_a, star_b)
    correct_label, fact = compute_correct_label(star_a, star_b, comparison)
    gemini_output = generate_comparative_question(
        client,
        star_a,
        star_b,
        df.loc[idx_a],  # raw data for star A
        df.loc[idx_b],  # raw data for star B
        comparison,
        correct_label,
        fact,
        max_retries=max_retries,
        delay=delay,
    )
    if gemini_output is None:
        return None

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return obj

    # Extract observational data for both stars (excluding derived parameters)
    excluded_cols = {
        'Teff', 'logg', 'FeH', 'obsid',
        'SpT', 'SpType', 'spectral_type', 'spec_type', 'spt', 
        'stage', 'evolutionary_stage', 'giant', 'dwarf', 'ms', 'mainsequence',
        'mass', 'radius', 'luminosity', 'age', 'distance', 'abs_mag',
        'M_V', 'M_K', 'M_J', 'M_H', 'vsini', 'v_rot', 'rotation', 'activity',
    }
    derived_keywords = ['est', 'pred', 'calc', 'derived', 'model', 'fit', 'interp']
    
    def extract_observational_data(raw_data: pd.Series) -> Dict[str, object]:
        obs_data = {}
        for col in raw_data.index:
            if col in excluded_cols or pd.isna(raw_data[col]):
                continue
            if any(keyword in col.lower() for keyword in derived_keywords):
                continue
            obs_data[col] = raw_data[col]
        return obs_data
    
    obs_data_a = extract_observational_data(df.loc[idx_a])
    obs_data_b = extract_observational_data(df.loc[idx_b])

    return convert_to_native({
        "pair_id": uuid.uuid4().hex,
        "indices": {"a": int(idx_a), "b": int(idx_b)},
        "obsids": {"a": star_a["obsid"], "b": star_b["obsid"]},
        "comparison_type": comparison,
        "question": gemini_output.get("Question"),
        "options": gemini_output.get("Options"),
        "model_answer_label": gemini_output.get("Answer"),
        "model_rationale": gemini_output.get("Rationale"),
        "expected_answer_label": correct_label,
        "expected_fact": fact,
        "star_a": star_a,
        "star_b": star_b,
        # Raw observational data for model training
        "observational_data_a": obs_data_a,
        "observational_data_b": obs_data_b,
    })


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [key for key in REQUIRED_DERIVED_PARAMS if key not in df.columns]
    if missing:
        raise ValueError(f"Input catalog missing required columns for answer generation: {missing}")
    
    # Define exclusion criteria (same as in format_star_block)
    excluded_cols = {
        'Teff', 'logg', 'FeH', 'obsid',
        'SpT', 'SpType', 'spectral_type', 'spec_type', 'spt', 
        'stage', 'evolutionary_stage', 'giant', 'dwarf', 'ms', 'mainsequence',
        'mass', 'radius', 'luminosity', 'age', 'distance', 'abs_mag',
        'M_V', 'M_K', 'M_J', 'M_H', 'vsini', 'v_rot', 'rotation', 'activity',
    }
    derived_keywords = ['est', 'pred', 'calc', 'derived', 'model', 'fit', 'interp']
    
    # Filter observational columns
    observational_cols = []
    excluded_by_name = []
    excluded_by_keyword = []
    
    for col in df.columns:
        if col in excluded_cols:
            excluded_by_name.append(col)
        elif any(keyword in col.lower() for keyword in derived_keywords):
            excluded_by_keyword.append(col)
        else:
            observational_cols.append(col)
    
    print(f"Columns available for questions: {observational_cols}")
    print(f"Excluded by name: {excluded_by_name}")
    print(f"Excluded by keywords: {excluded_by_keyword}")
    
    return df


def calculate_error_statistics(results: List[Dict[str, object]]) -> Dict[str, float]:
    """Calculate aggregate error statistics for parameter estimation"""
    teff_errors = []
    logg_errors = []
    teff_abs_errors = []
    logg_abs_errors = []
    
    for result in results:
        star_a = result["star_a"]
        star_b = result["star_b"]
        
        # Collect errors from both stars
        for star in [star_a, star_b]:
            if star["teff_error_k"] is not None:
                teff_errors.append(star["teff_error_k"])
                teff_abs_errors.append(star["teff_abs_error_k"])
            if star["logg_error"] is not None:
                logg_errors.append(star["logg_error"])
                logg_abs_errors.append(star["logg_abs_error"])
    
    # Convert to numpy arrays for calculations
    teff_errors = np.array(teff_errors)
    logg_errors = np.array(logg_errors)
    teff_abs_errors = np.array(teff_abs_errors)
    logg_abs_errors = np.array(logg_abs_errors)
    
    stats = {
        "num_teff_measurements": len(teff_errors),
        "num_logg_measurements": len(logg_errors),
        # Root Mean Square Error
        "teff_rmse_k": float(np.sqrt(np.mean(teff_errors**2))) if len(teff_errors) > 0 else None,
        "logg_rmse": float(np.sqrt(np.mean(logg_errors**2))) if len(logg_errors) > 0 else None,
        # Mean Absolute Error  
        "teff_mae_k": float(np.mean(teff_abs_errors)) if len(teff_abs_errors) > 0 else None,
        "logg_mae": float(np.mean(logg_abs_errors)) if len(logg_abs_errors) > 0 else None,
        # Standard deviation of errors
        "teff_std_k": float(np.std(teff_errors)) if len(teff_errors) > 0 else None,
        "logg_std": float(np.std(logg_errors)) if len(logg_errors) > 0 else None,
        # Mean bias (systematic offset)
        "teff_mean_bias_k": float(np.mean(teff_errors)) if len(teff_errors) > 0 else None,
        "logg_mean_bias": float(np.mean(logg_errors)) if len(logg_errors) > 0 else None,
        # Percentiles for error distribution
        "teff_p50_abs_error_k": float(np.percentile(teff_abs_errors, 50)) if len(teff_abs_errors) > 0 else None,
        "teff_p90_abs_error_k": float(np.percentile(teff_abs_errors, 90)) if len(teff_abs_errors) > 0 else None,
        "logg_p50_abs_error": float(np.percentile(logg_abs_errors, 50)) if len(logg_abs_errors) > 0 else None,
        "logg_p90_abs_error": float(np.percentile(logg_abs_errors, 90)) if len(logg_abs_errors) > 0 else None,
    }
    
    return stats


# Global variables for worker processes
_worker_df = None
_worker_evo = None
_worker_client = None

def init_worker(df_dict, evo_params, grid_data):
    """Initialize worker process with shared data"""
    global _worker_df, _worker_evo, _worker_client
    
    # Recreate DataFrame and EvolutionInference in worker process
    _worker_df = pd.DataFrame(df_dict)
    _worker_evo = EvolutionInference(**evo_params)
    
    # Load prebuilt grid data to avoid rebuilding in each worker
    if grid_data:
        _worker_evo.grid_points = grid_data['grid_points']
        _worker_evo.grid_teff = grid_data['grid_teff']
        _worker_evo.grid_logg = grid_data['grid_logg']
        _worker_evo.grid_luminosity = grid_data['grid_luminosity']
        _worker_evo.grid_radius = grid_data['grid_radius']
    
    # Each worker gets its own Gemini client
    _worker_client = init_gemini_client()

def process_pair_worker(args_tuple) -> Optional[Dict[str, object]]:
    """Worker function for multiprocessing - processes a single pair of stars"""
    (idx_a, idx_b, max_retries, delay) = args_tuple
    
    try:
        return process_pair(_worker_df, idx_a, idx_b, _worker_evo, _worker_client, max_retries, delay)
    except Exception as exc:
        print(f"Error processing pair ({idx_a}, {idx_b}): {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparative stellar MCQ dataset using Gemini")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input catalog (CSV)")
    parser.add_argument("--output-json", type=Path, required=True, help="Destination JSON file")
    parser.add_argument("--num-pairs", type=int, default=10000, help="How many star pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--grid-name", type=str, default="mist", help="Stellar evolution grid name")
    parser.add_argument(
        "--interp-method",
        type=str,
        default="nearest",
        choices=("nearest", "linear", "rbf", "kd_tree"),
        help="Interpolation method for stellar parameters",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Gemini retries per pair")
    parser.add_argument("--delay", type=float, default=0.25, help="Base delay between retries (seconds)")
    parser.add_argument(
        "--feh-range", 
        nargs=3, 
        type=float, 
        default=[-1.5, 0.8, 0.1], 
        metavar=("MIN", "MAX", "STEP"),
        help="Metallicity grid range: min max step (default: -1.5 0.8 0.1)"
    )
    parser.add_argument(
        "--num-processes", 
        type=int, 
        default=mp.cpu_count(), 
        help=f"Number of processes for parallel processing (default: {mp.cpu_count()})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_catalog(args.input_csv)
    
    # Create custom metallicity grid from command line args
    feh_min, feh_max, feh_step = args.feh_range
    feh_grid = np.arange(feh_min, feh_max + feh_step/2, feh_step)
    
    # Create EvolutionInference parameters for workers
    evo_params = {
        'grid_name': args.grid_name,
        'method': args.interp_method,
        'feh_grid': feh_grid
    }
    
    # Build grid once in main process
    evo = EvolutionInference(**evo_params)
    evo.build_dense_grid()

    pairs = sample_pairs(df, args.num_pairs, args.seed)
    
    # Convert DataFrame to dict for multiprocessing (DataFrames don't pickle well)
    df_dict = df.to_dict()
    
    # Extract grid data to pass to workers (avoid rebuilding grids)
    grid_data = {
        'grid_points': evo.grid_points,
        'grid_teff': evo.grid_teff,
        'grid_logg': evo.grid_logg,
        'grid_luminosity': evo.grid_luminosity,
        'grid_radius': evo.grid_radius,
    }
    
    # Prepare simplified arguments for worker processes (just the pair indices and params)
    worker_args = [
        (idx_a, idx_b, args.max_retries, args.delay)
        for idx_a, idx_b in pairs
    ]
    
    # Process pairs in parallel
    results: List[Dict[str, object]] = []
    
    if args.num_processes == 1:
        # Single process mode for debugging - initialize once
        init_worker(df_dict, evo_params, grid_data)
        for args_tuple in tqdm(worker_args, desc="Processing pairs", unit="pair"):
            record = process_pair_worker(args_tuple)
            if record is not None:
                results.append(record)
    else:
        # Multiprocessing mode
        with mp.Pool(
            processes=args.num_processes,
            initializer=init_worker,
            initargs=(df_dict, evo_params, grid_data)
        ) as pool:
            # Use imap with tqdm for progress tracking
            with tqdm(total=len(worker_args), desc="Processing pairs", unit="pair") as pbar:
                for result in pool.imap(process_pair_worker, worker_args):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
                    # Update description with success rate
                    success_rate = len(results) / pbar.n * 100
                    pbar.set_postfix({"Success": f"{success_rate:.1f}%", "Questions": len(results)})

    # Calculate aggregate error statistics
    error_stats = calculate_error_statistics(results)
    
    # Save results with error statistics
    output_data = {
        "questions": results,
        "error_statistics": error_stats,
        "metadata": {
            "num_questions": len(results),
            "grid_name": args.grid_name,
            "interp_method": args.interp_method,
            "feh_range": args.feh_range,
            "mass_grid_size": len(evo.mass_grid),
            "age_grid_size": len(evo.age_grid),
            "feh_grid_size": len(evo.feh_grid),
        }
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"Generated {len(results)} comparative questions and saved to {args.output_json}")
    print(f"Parameter error statistics:")
    print(f"  Teff RMSE: {error_stats['teff_rmse_k']:.1f} K")
    print(f"  Teff MAE: {error_stats['teff_mae_k']:.1f} K") 
    print(f"  log g RMSE: {error_stats['logg_rmse']:.3f}")
    print(f"  log g MAE: {error_stats['logg_mae']:.3f}")


if __name__ == "__main__":
    # Use 'fork' on Unix systems to avoid re-importing modules
    # Fall back to default on other platforms
    try:
        if hasattr(os, 'fork'):
            mp.set_start_method('fork')
        else:
            mp.set_start_method('spawn')
    except RuntimeError:
        # Already set, continue
        pass
    main()
