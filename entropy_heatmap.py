import os
import numpy as np
import librosa
import whisper
from whisper.audio import log_mel_spectrogram as whisper_log_mel_spectrogram
from whisper.audio import pad_or_trim as whisper_pad_or_trim
from whisper.audio import N_FRAMES as WHISPER_N_FRAMES
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from scipy.stats import entropy

warnings.filterwarnings("ignore")


class WhisperEntropyExtractor:
    """Class for extracting audio entropy features using Whisper"""

    def __init__(self, model_size="base", device=None):
        """
        Initialize Whisper model

        Parameters:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Computing device ("cpu", "cuda", "auto")
        """
        self.model_size = model_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Whisper model: {model_size}")
        print(f"Using device: {self.device}")

        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise

    def load_mono_audio(self, audio_path, target_sr=16000):
        """
        Load mono audio

        Parameters:
        audio_path: Audio file path
        target_sr: Target sampling rate (Whisper recommends 16000Hz)

        Returns:
        audio_data: Mono audio data
        sr: Sampling rate
        """
        try:
            # Load mono audio
            audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)

            if len(audio_data.shape) > 1:
                # If multi-channel, take the first channel
                audio_data = audio_data[0]

            return audio_data, sr

        except Exception as e:
            print(f"Failed to load audio file {audio_path}: {e}")
            return None, None

    def extract_whisper_features(self, audio_data):
        """
        Extract encoder features for the entire audio using Whisper

        Parameters:
        audio_data: 1D waveform (np.float32, 16kHz)

        Returns:
        features: Feature matrix (n_steps, feature_dim)
        timestamps: Timestamp for each time step (seconds)
        """
        try:
            # Ensure data type and shape
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.detach().cpu().numpy().astype(np.float32)
            else:
                audio_np = np.asarray(audio_data, dtype=np.float32)
            # Force to 1D (samples,)
            audio_np = np.squeeze(audio_np)
            if audio_np.ndim != 1:
                # If still not 1D, try to take the first dimension as time axis
                audio_np = np.reshape(audio_np, (-1,))
            if audio_np.size == 0:
                raise ValueError("empty audio after squeeze")

            # Calculate entire mel: (80, n_frames)
            with torch.no_grad():
                # Debug and robustness: ensure 1D, contiguous, float32
                audio_np = np.ascontiguousarray(audio_np, dtype=np.float32)
                # Directly call with torch tensor
                audio_t = torch.from_numpy(audio_np)
                if audio_t.ndim != 1:
                    raise ValueError(f"audio must be 1D, got shape {tuple(audio_t.shape)}")
                
                mel = whisper_log_mel_spectrogram(audio_t)  # (80, n_frames)
                # pad/trim to Whisper's required 3000 frames
                mel = whisper_pad_or_trim(mel, WHISPER_N_FRAMES)  # (80, 3000)
                # Add batch dimension: (1, 80, 3000)
                mel = mel.unsqueeze(0).to(self.device)

                # Pass through encoder: (1, n_frames, d_model)
                enc = self.model.encoder(mel)
                features = enc.squeeze(0).detach().cpu().numpy()  # (n_frames, d_model)

            # Whisper time step is about 10ms
            n_steps = features.shape[0]
            timestamps = (np.arange(n_steps, dtype=np.float32) * 0.01).tolist()

            return features, timestamps
        except Exception as e:
            print(f"Failed to extract Whisper features: {e}")
            return None, None

    def calculate_entropy_sum(self, audio_data):
        """
        Calculate the sum of model prediction entropy for all frames
        Use Whisper model to predict each frame and calculate prediction uncertainty

        Parameters:
        audio_data: 1D waveform (np.float32, 16kHz)

        Returns:
        entropy_sum: Sum of model prediction entropy for all frames
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                return None

            # Extract features for the entire audio
            features, timestamps = self.extract_whisper_features(audio_data)
            if features is None:
                return None

            # Calculate prediction entropy for each frame using model
            frame_entropies = []
            
            for i in range(len(features)):
                if i == 0:
                    # First frame: no previous context, entropy = 0
                    frame_entropies.append(0.0)
                    continue
                
                # Use model to predict current frame based on previous frames
                # Create input with previous frames as context
                context_frames = features[:i]  # All previous frames
                current_frame = features[i]    # Current frame to predict
                
                # Use Whisper model's decoder to predict current frame
                with torch.no_grad():
                    # Prepare input: previous frames as context
                    context_tensor = torch.from_numpy(context_frames).unsqueeze(0).to(self.device)  # (1, i, d_model)
                    
                    # Use model's decoder to predict next frame
                    # This is a simplified approach - in practice you'd need to adapt the model
                    # For now, we'll use the encoder output and calculate prediction uncertainty
                    
                    # Calculate prediction based on context
                    # Use attention mechanism or simple averaging
                    attention_weights = torch.softmax(torch.sum(context_tensor, dim=-1), dim=-1)  # (1, i)
                    prediction = torch.sum(context_tensor * attention_weights.unsqueeze(-1), dim=1)  # (1, d_model)
                    prediction = prediction.squeeze(0).cpu().numpy()  # (d_model,)
                    
                    # Calculate prediction error
                    prediction_error = current_frame - prediction
                    
                    # Calculate uncertainty as entropy of prediction error
                    # Convert error to probabilities using softmax
                    exp_error = np.exp(prediction_error - np.max(prediction_error))
                    probabilities = exp_error / np.sum(exp_error)
                    
                    # Calculate entropy of prediction uncertainty
                    frame_entropy = entropy(probabilities)
                    frame_entropies.append(frame_entropy)

            # Calculate sum of all frame prediction entropies
            entropy_sum = np.sum(frame_entropies)

            return entropy_sum

        except Exception as e:
            print(f"Failed to calculate model prediction entropy sum: {e}")
            return None

    def parse_stereo_filename(self, filename):
        """
        Parse stereo audio filename to extract left and right channel category information

        Parameters:
        filename: Audio filename, format: nn_left_category_left_id+right_category_right_id

        Returns:
        left_category: Left channel category
        right_category: Right channel category
        """
        try:
            # Remove file extension
            name_without_ext = os.path.splitext(filename)[0]

            # Check if starts with 'rn_' or 'nn_'
            if not (name_without_ext.startswith('rn_') or name_without_ext.startswith('nn_')):
                return None, None

            # Remove prefix ('rn_' or 'nn_')
            name_without_prefix = name_without_ext[3:]

            # Split left and right channels by '+'
            if '+' not in name_without_prefix:
                return None, None

            left_part, right_part = name_without_prefix.split('+', 1)

            # Extract left channel category (part before first underscore)
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None
            left_category = left_part[:left_underscore_pos]

            # Extract right channel category (part before first underscore)
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None
            right_category = right_part[:right_underscore_pos]

            return left_category, right_category

        except Exception as e:
            print(f"Failed to parse filename {filename}: {e}")
            return None, None

    def get_main_category(self, sub_category):
        """
        Get main category based on sub category

        Parameters:
        sub_category: Sub category name

        Returns:
        main_category: Main category name
        """
        category_mapping = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }
        for main_cat, sub_cats in category_mapping.items():
            if sub_category in sub_cats:
                return main_cat
        return 'Unknown'

    def process_stereo_entropy(self, input_path, output_dir, max_files=1000):
        """
        Process stereo audio files and calculate entropy sums for channel selection

        Parameters:
        input_path: Input directory path
        output_dir: Output directory path
        max_files: Maximum number of files to process

        Returns:
        results: Processing results list
        """
        print(f"\nProcessing stereo audio files for entropy analysis: {input_path}")
        print(f"Output directory: {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        # Load stereo audio files
        audio_path = Path(input_path)
        audio_files = list(audio_path.glob("*.wav"))

        if not audio_files:
            print(f"No matching audio files found in {input_path}")
            return all_results

        print(f"Found {len(audio_files)} audio files")

        # Randomly select up to max_files files
        if len(audio_files) > max_files:
            audio_files = random.sample(audio_files, max_files)
            print(f"Randomly selected {len(audio_files)} audio files for processing")
        else:
            print(f"Using all {len(audio_files)} audio files")

        for audio_file in tqdm(audio_files, desc="Processing stereo audio files"):
            try:
                # Load stereo audio
                audio_data, sr = librosa.load(str(audio_file), sr=16000, mono=False)

                if audio_data.ndim != 2 or audio_data.shape[0] != 2:
                    print(f"Skipping {audio_file.name}: Not stereo audio")
                    continue

                left, right = audio_data[0], audio_data[1]

                # Extract left and right channel features
                print(f"Extracting features for {audio_file.name}...")
                left_features, timestamps = self.extract_whisper_features(left)
                right_features, timestamps = self.extract_whisper_features(right)

                if left_features is None or right_features is None:
                    print(f"Skipping {audio_file.name}: Feature extraction failed")
                    continue

                # Calculate left and right channel entropy sums
                print(f"Calculating entropy sums for {audio_file.name}...")
                left_entropy_sum = self.calculate_entropy_sum(left)
                right_entropy_sum = self.calculate_entropy_sum(right)

                if left_entropy_sum is None or right_entropy_sum is None:
                    print(f"Skipping {audio_file.name}: Entropy sum calculation failed")
                    continue

                # Calculate left minus right difference
                left_minus_right_diff = left_entropy_sum - right_entropy_sum

                # Parse filename to get categories
                left_category, right_category = self.parse_stereo_filename(audio_file.name)

                # Determine which side to select (higher entropy value)
                if left_entropy_sum > right_entropy_sum:
                    selected_category = left_category
                    selected_side = 'Left'
                    entropy_value = left_entropy_sum
                    other_entropy_value = right_entropy_sum
                else:
                    selected_category = right_category
                    selected_side = 'Right'
                    entropy_value = right_entropy_sum
                    other_entropy_value = left_entropy_sum

                # Get main category
                main_category = self.get_main_category(selected_category) if selected_category else 'Unknown'

                # Build result
                result = {
                    'audio_file': audio_file.name,
                    'audio_path': str(audio_file),
                    'sampling_rate': sr,
                    'left_features_shape': left_features.shape,
                    'right_features_shape': right_features.shape,
                    'left_entropy_sum': left_entropy_sum,
                    'right_entropy_sum': right_entropy_sum,
                    'left_minus_right_diff': left_minus_right_diff,
                    'left_category': left_category,
                    'right_category': right_category,
                    'selected_category': selected_category,
                    'selected_side': selected_side,
                    'entropy_value': entropy_value,
                    'other_entropy_value': other_entropy_value,
                    'main_category': main_category,
                    'entropy_difference': abs(left_entropy_sum - right_entropy_sum),
                    'timestamps': timestamps,
                    'processing_time': datetime.now().isoformat()
                }
                all_results.append(result)

                print(f"  {audio_file.name}: Left entropy={left_entropy_sum:.4f}, Right entropy={right_entropy_sum:.4f}, Selected={selected_side}")

            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                continue

        # Plot heatmaps only
        if all_results:
            self.plot_entropy_heatmaps(all_results, output_dir)
        else:
            print("No audio files were successfully processed")

        return all_results


    def plot_entropy_heatmaps(self, results, output_dir):
        """
        Plot entropy-based selection heatmaps

        Parameters:
        results: Processing results list
        output_dir: Output directory
        """
        print("=== Creating Entropy-based Selection Heatmaps ===")

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Define color schemes for 5 main categories
        main_category_colors = {
            'High Ecology': '#FF6B6B',  # Red series
            'Low Ecology': '#DDA0DD',  # Purple series
            'speech': '#45B7D1',  # Blue series
            'music': '#96CEB4',  # Green series
            'Unknown Source': '#FFEAA7',  # Yellow series
        }

        # Only plot the main heatmap
        self.plot_entropy_pairing_heatmap(df, main_category_colors, output_dir)

    def plot_entropy_pairing_heatmap(self, df, main_category_colors, output_dir):
        """Plot category pairing entropy selection heatmap"""
        # Define standard category order
        target_categories = [
            'Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',  # High ecology
            'Computer keyboard', 'Helicopter', 'Chicken, rooster',  # Low ecology
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # Unknown source
        ]

        # Define main category groups
        main_category_groups = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }

        # Generate colors for each sub-category (based on main category)
        category_colors = {}
        for cat in target_categories:
            # Find the main category this sub-category belongs to
            main_cat = None
            for main_cat_name, sub_cats in main_category_groups.items():
                if cat in sub_cats:
                    main_cat = main_cat_name
                    break

            if main_cat:
                base_color = main_category_colors.get(main_cat, '#DDA0DD')
                # Adjust brightness based on position within the main category
                sub_cats = main_category_groups[main_cat]
                cat_index = sub_cats.index(cat)
                brightness = 0.4 + 0.6 * (cat_index / (len(sub_cats) - 1)) if len(sub_cats) > 1 else 0.7

                # Adjust color brightness
                import matplotlib.colors as mcolors
                base_rgb = mcolors.to_rgb(base_color)
                adjusted_color = tuple(c * brightness for c in base_rgb)
                category_colors[cat] = adjusted_color
            else:
                category_colors[cat] = '#DDA0DD'

        # Create pairing selection matrix (showing selected category colors)
        n_categories = len(target_categories)
        selection_matrix = np.full((n_categories, n_categories), np.nan)  # Use NaN to indicate no pairing
        color_matrix = np.full((n_categories, n_categories, 3), np.nan)  # RGB color matrix

        # Count selection results for each pairing
        for _, row in df.iterrows():
            left_cat = row['left_category']
            right_cat = row['right_category']
            selected_cat = row['selected_category']

            if left_cat in target_categories and right_cat in target_categories:
                left_idx = target_categories.index(left_cat)
                right_idx = target_categories.index(right_cat)

                # Get selected category color
                if selected_cat in category_colors:
                    selected_color = category_colors[selected_cat]
                    # Convert color to RGB values
                    if isinstance(selected_color, tuple):
                        color_matrix[left_idx, right_idx] = selected_color
                    else:
                        # If it's a color name, convert to RGB
                        rgb_color = mcolors.to_rgb(selected_color)
                        color_matrix[left_idx, right_idx] = rgb_color

                    # Set an identifier value to indicate pairing exists
                    selection_matrix[left_idx, right_idx] = 1

        # Draw heatmap
        fig, ax = plt.subplots(1, 1, figsize=(18, 16))

        # Create custom color mapping
        # First draw background (white)
        ax.imshow(np.ones((n_categories, n_categories)), cmap='gray', vmin=0, vmax=1, alpha=0.3)

        # Draw color for each cell
        for i in range(n_categories):
            for j in range(n_categories):
                if not np.isnan(color_matrix[i, j, 0]):  # If there is color data
                    # Draw colored rectangle
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=color_matrix[i, j],
                                         edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

        # Set labels
        ax.set_xticks(range(n_categories))
        ax.set_yticks(range(n_categories))

        # Set X-axis labels (right channel categories)
        x_labels = ax.set_xticklabels(target_categories, rotation=45, ha='right', fontsize=9)
        for i, label in enumerate(x_labels):
            label.set_color(category_colors[target_categories[i]])
            label.set_fontweight('bold')

        # Set Y-axis labels (left channel categories)
        y_labels = ax.set_yticklabels(target_categories, fontsize=9)
        for i, label in enumerate(y_labels):
            label.set_color(category_colors[target_categories[i]])
            label.set_fontweight('bold')

        # Add category name annotations
        for i in range(n_categories):
            for j in range(n_categories):
                if not np.isnan(color_matrix[i, j, 0]):  # If there is pairing
                    # Get the category selected for this pairing
                    for _, row in df.iterrows():
                        left_cat = row['left_category']
                        right_cat = row['right_category']
                        if (left_cat == target_categories[i] and
                                right_cat == target_categories[j]):
                            selected_cat = row['selected_category']
                            # Display selected category name (simplified version)
                            short_name = selected_cat.split(',')[0]  # Take part before comma
                            ax.text(j, i, short_name, ha='center', va='center',
                                    fontweight='bold', color='white', fontsize=6)

        ax.set_title('Entropy-based Category Pairing Selection Results\n(Color = Selected Category)',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Right Channel Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Left Channel Category', fontsize=12, fontweight='bold')

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, n_categories, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_categories, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

        # Set axis range
        ax.set_xlim(-0.5, n_categories - 0.5)
        ax.set_ylim(-0.5, n_categories - 0.5)

        plt.tight_layout()

        # Save chart
        heatmap_path = os.path.join(output_dir, "entropy_pairing_selection_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Entropy pairing selection heatmap saved: {heatmap_path}")



def main():
    """Main function"""
    print("=== Whisper Entropy-based Channel Selection Tool ===")

    # Set parameters
    model_size = "base"  # Options: "tiny", "base", "small", "medium", "large"

    # Set paths
    input_dir = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"  # Original audio directory
    output_dir = r"D:\D\research\audioset下载\entropy_analysis"  # Output directory

    # Create entropy extractor
    extractor = WhisperEntropyExtractor(model_size=model_size)

    # Process stereo audio files for entropy analysis
    results = extractor.process_stereo_entropy(
        input_path=input_dir,
        output_dir=output_dir,
        max_files=1000)  # Set a large number to ensure processing all files

    print(f"\nEntropy analysis completed, results saved to: {output_dir}")


if __name__ == "__main__":
    main()
