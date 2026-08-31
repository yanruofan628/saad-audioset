import os
import numpy as np
import librosa
import soundfile as sf
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

warnings.filterwarnings("ignore")


class WhisperStereoFeatureExtractor:
    """Class for extracting audio features using Whisper (supports mono and stereo)"""

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

    def extract_whisper_features(self, audio_data, frame_duration_ms=8):
        """
        Extract encoder features for the entire audio using Whisper, returning at Whisper's native time steps (~10ms).

        Parameters:
        audio_data: 1D waveform (np.float32, 16kHz)
        frame_duration_ms: Reserved parameter (ignored), time step fixed at ~10ms

        Returns:
        features: Feature matrix (n_steps, feature_dim)
        timestamps: Timestamp for each time step (seconds), step length ~0.01s
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
                # Record shape and dtype
                # print(f"[DEBUG] audio shape: {tuple(audio_t.shape)}, dtype: {audio_t.dtype}")
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

    def extract_librosa_features(self, audio_data, frame_duration_ms=8):
        """
        Extract MFCC features using librosa as an alternative approach

        Parameters:
        audio_data: Audio data
        frame_duration_ms: Frame length (milliseconds)

        Returns:
        features: Extracted feature matrix (n_frames, feature_dim)
        timestamps: Timestamp list
        """
        try:
            sr = 16000
            frame_length = int(sr * frame_duration_ms / 1000)
            total_frames = len(audio_data) // frame_length

            if total_frames == 0:
                print("Warning: Audio too short, cannot extract features")
                return None, None

            features_list = []
            timestamps = []

            print(f"Using librosa to extract MFCC features, total frames: {total_frames}")

            # Extract MFCC features frame by frame
            for i in tqdm(range(total_frames), desc="Extracting librosa features"):
                start_idx = i * frame_length
                end_idx = start_idx + frame_length
                frame_audio = audio_data[start_idx:end_idx]

                # Ensure frame length is correct
                if len(frame_audio) < frame_length:
                    frame_audio = np.pad(frame_audio, (0, frame_length - len(frame_audio)), 'constant')

                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=frame_audio, sr=sr, n_mfcc=13)
                frame_features = np.mean(mfcc, axis=1)  # Average over time dimension

                features_list.append(frame_features)
                timestamps.append(i * frame_duration_ms / 1000)

            return np.array(features_list), timestamps

        except Exception as e:
            print(f"Failed to extract librosa features: {e}")
            return None, None

    def calculate_feature_variation_sum(self, left_features, right_features):
        """
        Calculate the sum of feature variations for all frames in left and right channels

        Parameters:
        left_features: Left channel feature matrix (n_frames, feature_dim)
        right_features: Right channel feature matrix (n_frames, feature_dim)

        Returns:
        left_variation_sum: Left channel frame variation sum
        right_variation_sum: Right channel frame variation sum
        left_minus_right_diff: Difference between left and right channel variation sums
        variation_details: Variation details for each frame
        """
        try:
            # Ensure both feature matrices have consistent shapes
            min_frames = min(len(left_features), len(right_features))
            left_features = left_features[:min_frames]
            right_features = right_features[:min_frames]

            # Calculate left channel frame variations (L2 distance from previous frame)
            left_frame_variations = []
            for i in range(min_frames):
                if i == 0:
                    # First frame, variation is 0
                    left_frame_variations.append(0.0)
                else:
                    # Calculate L2 distance from previous frame
                    left_variation = np.linalg.norm(left_features[i] - left_features[i - 1])
                    left_frame_variations.append(left_variation)

            # Calculate right channel frame variations (L2 distance from previous frame)
            right_frame_variations = []
            for i in range(min_frames):
                if i == 0:
                    # First frame, variation is 0
                    right_frame_variations.append(0.0)
                else:
                    # Calculate L2 distance from previous frame
                    right_variation = np.linalg.norm(right_features[i] - right_features[i - 1])
                    right_frame_variations.append(right_variation)

            # Calculate left and right channel frame variation sums
            left_variation_sum = np.sum(left_frame_variations)
            right_variation_sum = np.sum(right_frame_variations)

            # Calculate left minus right difference
            left_minus_right_diff = left_variation_sum - right_variation_sum

            # Calculate statistics
            variation_stats = {
                'left_variation_sum': left_variation_sum,
                'right_variation_sum': right_variation_sum,
                'left_minus_right_diff': left_minus_right_diff,
                'mean_left_variation': np.mean(left_frame_variations[1:]),  # Exclude first frame
                'mean_right_variation': np.mean(right_frame_variations[1:]),  # Exclude first frame
                'std_left_variation': np.std(left_frame_variations[1:]),
                'std_right_variation': np.std(right_frame_variations[1:]),
                'max_left_variation': np.max(left_frame_variations[1:]),
                'max_right_variation': np.max(right_frame_variations[1:]),
                'min_left_variation': np.min(left_frame_variations[1:]),
                'min_right_variation': np.min(right_frame_variations[1:]),
                'frame_count': min_frames
            }

            return left_variation_sum, right_variation_sum, left_minus_right_diff, left_frame_variations, right_frame_variations, variation_stats

        except Exception as e:
            print(f"Failed to calculate feature variations: {e}")
            return None, None, None, None, None, None

    def process_audio_pair(self, input_path, out_path, frame_duration_ms=8, file_pattern='*.wav', max_files=20):
        """
        Process stereo audio files, extract left and right channel features and calculate left minus right feature variation sum

        Parameters:
        input_path: Input directory path
        out_path: Output directory path
        frame_duration_ms: Frame length (milliseconds)
        file_pattern: File matching pattern
        max_files: Maximum number of files to process (random selection)

        Returns:
        all_results: Processing results list
        """
        print(f"\nProcessing stereo audio files directory: {input_path}")
        print(f"Output directory: {out_path}")

        # Create output directory
        os.makedirs(out_path, exist_ok=True)

        all_results = []
        # Load stereo audio files
        audio_path = Path(input_path)
        audio_files = list(audio_path.glob(file_pattern))

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
                left_features, timestamps = self.extract_whisper_features(left, frame_duration_ms)
                right_features, timestamps = self.extract_whisper_features(right, frame_duration_ms)

                if left_features is None or right_features is None:
                    print(f"Skipping {audio_file.name}: Feature extraction failed")
                    continue

                # Calculate left and right channel variation sums
                print(f"Calculating variation sums for {audio_file.name}...")
                left_variation_sum = self.calculate_mono_variation_sum(left_features)
                right_variation_sum = self.calculate_mono_variation_sum(right_features)

                if left_variation_sum is None or right_variation_sum is None:
                    print(f"Skipping {audio_file.name}: Variation sum calculation failed")
                    continue

                # Calculate left minus right difference
                left_minus_right_diff = left_variation_sum - right_variation_sum

                # Build result
                result = {
                    'audio_file': audio_file.name,
                    'audio_path': str(audio_file),
                    'frame_duration_ms': frame_duration_ms,
                    'sampling_rate': sr,
                    'left_features_shape': left_features.shape,
                    'right_features_shape': right_features.shape,
                    'left_variation_sum': left_variation_sum,
                    'right_variation_sum': right_variation_sum,
                    'left_minus_right_diff': left_minus_right_diff,
                    'timestamps': timestamps,
                    'processing_time': datetime.now().isoformat()
                }
                all_results.append(result)

                print(
                    f"  {audio_file.name}: Left variation={left_variation_sum:.4f}, Right variation={right_variation_sum:.4f}, Difference={left_minus_right_diff:.4f}")

            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                continue

        # Save results and plot histograms
        if all_results:
            self.save_stereo_results(all_results, out_path)
            self.plot_stereo_histograms(all_results, out_path)
        else:
            print("No audio files were successfully processed")

        return all_results

    def save_stereo_results(self, results, output_dir):
        """
        保存立体声处理结果

        参数:
        results: 处理结果列表
        output_dir: 输出目录
        """
        # 保存汇总结果（CSV格式）
        csv_data = []
        all_left_minus_right_diffs = []

        for result in results:
            csv_data.append({
                'audio_file': result['audio_file'],
                'frame_duration_ms': result['frame_duration_ms'],
                'sampling_rate': result['sampling_rate'],
                'left_features_shape': str(result['left_features_shape']),
                'right_features_shape': str(result['right_features_shape']),
                'left_variation_sum': float(result['left_variation_sum']),
                'right_variation_sum': float(result['right_variation_sum']),
                'left_minus_right_diff': float(result['left_minus_right_diff']),
                'processing_time': result['processing_time']
            })

            # 收集数据用于直方图
            all_left_minus_right_diffs.append(float(result['left_minus_right_diff']))

        # 保存CSV文件
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, "stereo_features_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存numpy数组
        np_path = os.path.join(output_dir, "stereo_variation_data.npz")
        np.savez(np_path,
                 left_minus_right_diffs=np.array(all_left_minus_right_diffs),
                 csv_data=df.values,
                 column_names=df.columns.tolist())

        print(f"Stereo results saved:")
        print(f"  Summary results: {csv_path}")
        print(f"  Numpy data: {np_path}")

    def plot_stereo_histograms(self, results, output_dir):
        """
        绘制立体声左减右表征变化和的直方图

        参数:
        results: 处理结果列表
        output_dir: 输出目录
        """
        # 提取左减右差值数据
        left_minus_right_diffs = [float(r['left_minus_right_diff']) for r in results]

        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')

        # 创建单个直方图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Left Minus Right Variation Sum Distribution', fontsize=16, fontweight='bold')

        # 绘制左减右差值分布直方图
        ax.hist(left_minus_right_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of (Left - Right) Variation Sum', fontsize=14)
        ax.set_xlabel('Left - Right Variation Sum Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"""
        Statistics:
        Audio Files: {len(results)}

        Left - Right Differences:
        Mean: {np.mean(left_minus_right_diffs):.4f}
        Std: {np.std(left_minus_right_diffs):.4f}
        Min: {np.min(left_minus_right_diffs):.4f}
        Max: {np.max(left_minus_right_diffs):.4f}
        Median: {np.median(left_minus_right_diffs):.4f}
        """

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor="lightgray", alpha=0.8), fontsize=10)

        plt.tight_layout()

        # 保存图片
        hist_path = os.path.join(output_dir, "left_minus_right_histogram.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nLeft-Right Variation Sum Histogram Statistics:")
        print(f"  Processed audio files: {len(results)}")
        print(
            f"  Left-Right difference - Mean: {np.mean(left_minus_right_diffs):.4f}, Std: {np.std(left_minus_right_diffs):.4f}")
        print(f"  Min: {np.min(left_minus_right_diffs):.4f}, Max: {np.max(left_minus_right_diffs):.4f}")
        print(f"  Median: {np.median(left_minus_right_diffs):.4f}")
        print(f"  Histogram saved: {hist_path}")

    def process_audio_file(self, audio_path, frame_duration_ms=8):
        """
        处理单个音频文件，提取单声道特征并计算变化

        参数:
        audio_path: 音频文件路径
        frame_duration_ms: 帧长度（毫秒）

        返回:
        result: 处理结果字典
        """
        print(f"\nProcessing audio file: {os.path.basename(audio_path)}")

        # Load mono audio
        audio_data, sr = self.load_mono_audio(audio_path)
        if audio_data is None:
            return None

        # Extract features
        print("Extracting audio features...")
        features, timestamps = self.extract_whisper_features(audio_data, frame_duration_ms)
        if features is None:
            return None

        # Calculate mono frame variation sum
        print("Calculating frame variation sum...")
        variation_sum = self.calculate_mono_variation_sum(features)
        if variation_sum is None:
            return None

        # 构建结果
        result = {
            'audio_file': os.path.basename(audio_path),
            'audio_path': audio_path,
            'frame_duration_ms': frame_duration_ms,
            'sampling_rate': sr,
            'features_shape': features.shape,
            'variation_sum': variation_sum,
            'timestamps': timestamps,
            'processing_time': datetime.now().isoformat()
        }

        print(f"Processing completed - Frame variation sum: {variation_sum:.4f}")

        return result

    def process_directory(self, input_dir, output_dir, frame_duration_ms=8, file_pattern="*.wav"):
        """
        批量处理目录中的音频文件（只处理指定的类别）

        参数:
        input_dir: 输入目录
        output_dir: 输出目录
        frame_duration_ms: 帧长度（毫秒）
        file_pattern: 文件匹配模式
        """
        print(f"Starting batch processing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Frame duration: {frame_duration_ms}ms")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define specific categories to process
        target_categories = [
            'Alarm', 'Crying, sobbing', 'Roar',  # High ecology
            'Computer keyboard', 'Fireworks', 'Chicken, rooster',  # Low ecology
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # Unknown source
        ]

        print(f"Target categories: {target_categories}")

        # Organize files by category
        category_files = {}
        total_files = 0

        for category in target_categories:
            category_path = Path(input_dir) / category
            if category_path.exists() and category_path.is_dir():
                # Find audio files in this category directory
                audio_files = list(category_path.glob(file_pattern))
                if audio_files:
                    category_files[category] = audio_files
                    total_files += len(audio_files)
                    print(f"  {category}: {len(audio_files)} files")
                else:
                    print(f"  {category}: No audio files found")
            else:
                print(f"  {category}: Directory does not exist")

        if total_files == 0:
            print(f"No matching audio files found in specified categories")
            return

        print(f"Total found {total_files} audio files")

        # Processing results list
        all_results = []

        # Process audio files by category
        for category, files in category_files.items():
            print(f"\nProcessing category: {category}")
            for audio_file in tqdm(files, desc=f"Processing {category}"):
                result = self.process_audio_file(str(audio_file), frame_duration_ms)
                if result is not None:
                    # Add category information to results
                    result['category'] = category
                    all_results.append(result)

        # Save results
        self.save_results(all_results, output_dir)

        print(f"\nBatch processing completed, processed {len(all_results)} files")

    def save_results(self, results, output_dir):
        """
        保存处理结果并绘制直方图

        参数:
        results: 处理结果列表
        output_dir: 输出目录
        """
        # 保存汇总结果（CSV格式）
        csv_data = []
        all_variation_sums = []

        for result in results:
            csv_data.append({
                'category': result.get('category', 'Unknown'),  # 添加类别信息
                'audio_file': result['audio_file'],
                'frame_duration_ms': result['frame_duration_ms'],
                'sampling_rate': result['sampling_rate'],
                'features_shape': str(result['features_shape']),
                'variation_sum': float(result['variation_sum']),
                'processing_time': result['processing_time']
            })

            # 收集数据用于直方图
            all_variation_sums.append(float(result['variation_sum']))

        # 保存CSV文件
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, "whisper_mono_features_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存numpy数组
        np_path = os.path.join(output_dir, "variation_data.npz")
        np.savez(np_path,
                 variation_sums=np.array(all_variation_sums),
                 csv_data=df.values,
                 column_names=df.columns.tolist())

        # 绘制直方图
        self.plot_histograms(results, output_dir)

        print(f"Results saved:")
        print(f"  Summary results: {csv_path}")
        print(f"  Numpy data: {np_path}")
        print(f"  Histogram: {output_dir}/variation_histograms.png")

    def plot_histograms(self, results, output_dir):
        """
        绘制帧变化和的直方图

        参数:
        results: 处理结果列表
        output_dir: 输出目录
        """
        # 提取数据
        variation_sums = [float(r['variation_sum']) for r in results]

        # 按类别组织数据
        category_data = {}
        for result in results:
            category = result.get('category', 'Unknown')
            if category not in category_data:
                category_data[category] = {
                    'variation_sums': []
                }
            category_data[category]['variation_sums'].append(float(result['variation_sum']))
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mono Audio Feature Variation Statistics by Category', fontsize=16, fontweight='bold')

        # 1. 整体帧变化和分布直方图
        axes[0, 0].hist(variation_sums, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Frame Variation Sum Distribution', fontsize=14)
        axes[0, 0].set_xlabel('Variation Sum Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 按类别的帧变化和分布直方图（叠加显示）
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_data)))
        for i, (category, data) in enumerate(category_data.items()):
            if data['variation_sums']:
                axes[0, 1].hist(data['variation_sums'], bins=20, alpha=0.6,
                                color=colors[i], edgecolor='black',
                                label=f'{category} (n={len(data["variation_sums"])})')
        axes[0, 1].set_title('Frame Variation Sum Distribution by Category', fontsize=14)
        axes[0, 1].set_xlabel('Variation Sum Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 按类别的帧变化和箱线图
        if len(category_data) > 1:
            category_names = list(category_data.keys())
            category_variation_sums = [category_data[cat]['variation_sums'] for cat in category_names]
            bp1 = axes[1, 0].boxplot(category_variation_sums, patch_artist=True, labels=category_names)
            for i, patch in enumerate(bp1['boxes']):
                patch.set_facecolor(colors[i])
            axes[1, 0].set_title('Frame Variation Sum by Category (Box Plot)', fontsize=14)
            axes[1, 0].set_ylabel('Variation Sum Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 如果只有一个类别，显示整体分布
            axes[1, 0].boxplot(variation_sums, patch_artist=True,
                               boxprops=dict(facecolor='skyblue', alpha=0.7))
            axes[1, 0].set_title('Frame Variation Sum Box Plot', fontsize=14)
        axes[1, 0].set_ylabel('Variation Sum Value')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 类别统计对比（柱状图）
        category_means = []
        category_stds = []
        category_names = []
        for category, data in category_data.items():
            if data['variation_sums']:
                category_means.append(np.mean(data['variation_sums']))
                category_stds.append(np.std(data['variation_sums']))
                category_names.append(category)

        if category_means:
            x_pos = np.arange(len(category_names))
            bars = axes[1, 1].bar(x_pos, category_means, yerr=category_stds,
                                  capsize=5, alpha=0.7, color=colors[:len(category_names)])
            axes[1, 1].set_title('Mean Frame Variation Sum by Category', fontsize=14)
            axes[1, 1].set_ylabel('Mean Variation Sum Value')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(category_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"""
        Statistics:
        Audio Files: {len(results)}
        Categories: {len(category_data)}

        Overall Variation Sums:
        Mean: {np.mean(variation_sums):.4f}
        Std: {np.std(variation_sums):.4f}
        Min: {np.min(variation_sums):.4f}
        Max: {np.max(variation_sums):.4f}
        """

        fig.text(0.02, 0.02, stats_text, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # 保存图片
        hist_path = os.path.join(output_dir, "variation_histograms.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nHistogram Statistics:")
        print(f"  Processed Audio Files: {len(results)}")
        print(f"  Categories: {len(category_data)}")
        print(f"  Overall Variation Sums - Mean: {np.mean(variation_sums):.4f}, Std: {np.std(variation_sums):.4f}")

        # 打印每个类别的统计信息
        for category, data in category_data.items():
            if data['variation_sums']:
                print(f"  {category}: {len(data['variation_sums'])} files, Mean: {np.mean(data['variation_sums']):.4f}")

    def calculate_category_frame_variations(self, input_dir, output_dir):
        """
        计算每个类别音频的帧变化和分布

        参数:
        input_dir: 原始音频目录
        output_dir: 输出目录
        """
        print("=== 开始计算每个类别的帧变化和分布 ===")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")

        # 定义15个子类别
        category_mapping = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
        }

        print("直接从原始音频文件处理，无需预存特征数据")

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 输入目录不存在: {input_dir}")
            return

        # 收集每个类别的帧变化和
        category_variations = {}
        all_variations = []

        for main_cat, sub_cats in category_mapping.items():
            print(f"\n处理类别: {main_cat}")
            category_variations[main_cat] = {}

            for sub_cat in sub_cats:
                print(f"  处理子类别: {sub_cat}")
                sub_cat_variations = []

                # 构建子类别目录路径
                sub_cat_dir = os.path.join(input_dir, sub_cat)
                print(f"    检查目录: {sub_cat_dir}")
                if not os.path.exists(sub_cat_dir):
                    print(f"    跳过 {sub_cat}: 目录不存在")
                    continue

                # 查找该目录下的音频文件
                audio_files = list(Path(sub_cat_dir).glob("*.wav"))
                print(f"    找到 {len(audio_files)} 个音频文件")
                if not audio_files:
                    print(f"    跳过 {sub_cat}: 未找到音频文件")
                    continue

                for audio_file in tqdm(audio_files, desc=f"处理{sub_cat}"):
                    # 加载单声道音频
                    try:
                        audio_data, sr = librosa.load(str(audio_file), sr=16000, mono=True)
                        if audio_data is None or len(audio_data) == 0:
                            print(f"    音频数据为空: {audio_file.name}")
                            continue
                    except Exception as e:
                        print(f"    加载音频失败 {audio_file.name}: {e}")
                        continue

                    # 提取Whisper特征
                    features, _ = self.extract_whisper_features(audio_data)
                    if features is None:
                        print(f"    特征提取失败: {audio_file.name}")
                        continue

                    # 计算单声道帧变化和
                    variation_sum = self.calculate_mono_variation_sum(features)
                    if variation_sum is None:
                        print(f"    变化和计算失败: {audio_file.name}")
                        continue

                    # 添加结果
                    sub_cat_variations.append({
                        'file_name': audio_file.name,
                        'variation_sum': variation_sum
                    })
                    all_variations.append({
                        'main_category': main_cat,
                        'sub_category': sub_cat,
                        'file_name': audio_file.name,
                        'variation_sum': variation_sum
                    })

                category_variations[main_cat][sub_cat] = sub_cat_variations
                print(f"    {sub_cat}: 处理了 {len(sub_cat_variations)} 个音频文件")

        # 绘制分布图
        self.plot_category_variations(category_variations, all_variations, output_dir)

        # 保存结果
        self.save_category_variations(category_variations, all_variations, output_dir)

        print(f"\n类别帧变化和分析完成，结果保存到: {output_dir}")

    def calculate_mono_variation_sum(self, features):
        """
        计算单声道音频的帧变化和

        参数:
        features: 特征矩阵 (n_frames, feature_dim)

        返回:
        variation_sum: 帧变化和
        """
        try:
            if features is None or len(features) == 0:
                return None

            # 计算每帧的变化（与前一帧的L2距离）
            frame_variations = []
            for i in range(len(features)):
                if i == 0:
                    # 第一帧，变化为0
                    frame_variations.append(0.0)
                else:
                    # 计算与前一帧的L2距离
                    variation = np.linalg.norm(features[i] - features[i - 1])
                    frame_variations.append(variation)

            # 计算所有帧的变化和
            variation_sum = np.sum(frame_variations)

            return variation_sum

        except Exception as e:
            print(f"计算单声道帧变化和失败: {e}")
            return None

    def plot_category_variations(self, category_variations, all_variations, output_dir):
        """
        绘制每个类别的帧变化和分布图

        参数:
        category_variations: 按类别组织的帧变化数据
        all_variations: 所有音频的帧变化数据
        output_dir: 输出目录
        """
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')

        # 创建子类别和主类别的颜色映射
        sub_categories = []
        for main_cat_data in category_variations.values():
            for sub_cat in main_cat_data.keys():
                sub_categories.append(sub_cat)

        main_categories = list(category_variations.keys())
        sub_colors = plt.cm.Set3(np.linspace(0, 1, len(sub_categories)))
        main_colors = plt.cm.Set1(np.linspace(0, 1, len(main_categories)))

        # 1. 为每个子类别（15个）绘制独立的直方图
        print("正在绘制子类别直方图...")
        for i, sub_cat in enumerate(sub_categories):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # 找到该子类别对应的数据
            variations = []
            main_cat_name = ""
            for main_cat, sub_cats_data in category_variations.items():
                if sub_cat in sub_cats_data and sub_cats_data[sub_cat]:
                    variations = [d['variation_sum'] for d in sub_cats_data[sub_cat]]
                    main_cat_name = main_cat
                    break

            if variations:
                ax.hist(variations, bins=20, alpha=0.7, color=sub_colors[i],
                        edgecolor='black', label=f'{sub_cat} (n={len(variations)})')
                ax.set_title(f'Variation Sum Distribution - {sub_cat}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Variation Sum Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = f'Main Category: {main_cat_name}\nMean: {np.mean(variations):.2f}\nStd: {np.std(variations):.2f}\nMin: {np.min(variations):.2f}\nMax: {np.max(variations):.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                           facecolor="lightgray", alpha=0.8), fontsize=10)
            else:
                # 如果没有数据，显示提示信息
                ax.text(0.5, 0.5, f'No data for {sub_cat}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'Variation Sum Distribution - {sub_cat} (No Data)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Variation Sum Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存每个子类别的图
            safe_sub_cat_name = sub_cat.replace(',', '_').replace(' ', '_').replace('(', '').replace(')', '')
            hist_path = os.path.join(output_dir, f"subcategory_histogram_{safe_sub_cat_name}.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存

            print(f"  子类别 {sub_cat} 直方图已保存: {hist_path}")

        # 2. 为每个主类别（5个）绘制独立的直方图
        print("正在绘制主类别直方图...")
        for i, main_cat in enumerate(main_categories):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            variations = []
            for sub_cat_data in category_variations[main_cat].values():
                for file_data in sub_cat_data:
                    variations.append(file_data['variation_sum'])

            if variations:
                ax.hist(variations, bins=20, alpha=0.7, color=main_colors[i],
                        edgecolor='black', label=f'{main_cat} (n={len(variations)})')
                ax.set_title(f'Variation Sum Distribution - {main_cat}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Variation Sum Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = f'Mean: {np.mean(variations):.2f}\nStd: {np.std(variations):.2f}\nMin: {np.min(variations):.2f}\nMax: {np.max(variations):.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                           facecolor="lightgray", alpha=0.8), fontsize=10)
            else:
                # 如果没有数据，显示提示信息
                ax.text(0.5, 0.5, f'No data for {main_cat}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'Variation Sum Distribution - {main_cat} (No Data)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Variation Sum Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存每个主类别的图
            safe_main_cat_name = main_cat.replace(' ', '_')
            hist_path = os.path.join(output_dir, f"maincategory_histogram_{safe_main_cat_name}.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存

            print(f"  主类别 {main_cat} 直方图已保存: {hist_path}")

        # 3. 绘制整体分布图
        print("正在绘制整体分布图...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        all_variations_list = [v['variation_sum'] for v in all_variations]

        # 检查是否有数据
        if not all_variations_list:
            print("警告: 没有找到任何变化数据，跳过整体分布图绘制")
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='red')
            ax.set_title('Overall Variation Sum Distribution - No Data', fontsize=14, fontweight='bold')
        else:
            ax.hist(all_variations_list, bins=30, alpha=0.7, color='skyblue',
                    edgecolor='black', label=f'All Categories (n={len(all_variations_list)})')
            ax.set_title('Overall Variation Sum Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Variation Sum Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            stats_text = f'Mean: {np.mean(all_variations_list):.2f}\nStd: {np.std(all_variations_list):.2f}\nMin: {np.min(all_variations_list):.2f}\nMax: {np.max(all_variations_list):.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                       facecolor="lightgray", alpha=0.8), fontsize=10)

        plt.tight_layout()

        # 保存整体分布图
        hist_path = os.path.join(output_dir, "overall_variation_distribution.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"整体分布图已保存: {hist_path}")

        # 4. 子类别箱线图对比
        print("正在绘制子类别箱线图...")
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        sub_cat_variations = []
        for main_cat_data in category_variations.values():
            for sub_cat, data in main_cat_data.items():
                if data:
                    sub_cat_variations.append([d['variation_sum'] for d in data])

        if sub_cat_variations:
            bp = ax.boxplot(sub_cat_variations, patch_artist=True, labels=sub_categories)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(sub_colors[i])
            ax.set_title('Variation Sum by Sub-category (Box Plot)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Variation Sum Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存子类别箱线图
        box_path = os.path.join(output_dir, "subcategory_boxplot.png")
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"子类别箱线图已保存: {box_path}")

        # 5. 主类别箱线图对比
        print("正在绘制主类别箱线图...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        main_cat_stats = []
        main_cat_labels = []
        for main_cat in main_categories:
            variations = []
            for sub_cat_data in category_variations[main_cat].values():
                for file_data in sub_cat_data:
                    variations.append(file_data['variation_sum'])

            if variations:
                main_cat_stats.append(variations)
                main_cat_labels.append(f'{main_cat}\n(n={len(variations)})')

        if main_cat_stats:
            bp2 = ax.boxplot(main_cat_stats, patch_artist=True, labels=main_cat_labels)
            for i, patch in enumerate(bp2['boxes']):
                patch.set_facecolor(main_colors[i])
            ax.set_title('Variation Sum by Main Category (Box Plot)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Variation Sum Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存主类别箱线图
        main_box_path = os.path.join(output_dir, "maincategory_boxplot.png")
        plt.savefig(main_box_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"主类别箱线图已保存: {main_box_path}")

        print(f"\n所有图表绘制完成！")
        print(f"  - 子类别直方图: 15张")
        print(f"  - 主类别直方图: 5张")
        print(f"  - 整体分布图: 1张")
        print(f"  - 箱线图: 2张")
        print(f"  - 总计: 23张图表")

    def parse_stereo_filename(self, filename):
        """
        解析立体声音频文件名，提取左右声道类别信息

        参数:
        filename: 音频文件名，格式为 nn_左声道类别_左声道ID+右声道类别_右声道ID

        返回:
        left_category: 左声道类别
        right_category: 右声道类别
        """
        try:
            # 移除文件扩展名
            name_without_ext = os.path.splitext(filename)[0]

            # 检查是否以 'rn_' 或 'nn_' 开头
            if not (name_without_ext.startswith('rn_') or name_without_ext.startswith('nn_')):
                return None, None

            # 移除前缀（'rn_' 或 'nn_'）
            name_without_prefix = name_without_ext[3:]

            # 按 '+' 分割左右声道
            if '+' not in name_without_prefix:
                return None, None

            left_part, right_part = name_without_prefix.split('+', 1)

            # 提取左声道类别（第一个下划线前的部分）
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None
            left_category = left_part[:left_underscore_pos]  # 第一个下划线前的部分

            # 提取右声道类别（第一个下划线前的部分）
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None
            right_category = right_part[:right_underscore_pos]  # 第一个下划线前的部分

            return left_category, right_category

        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None

    def get_main_category(self, sub_category):
        """
        根据子类别获取主类别

        参数:
        sub_category: 子类别名称

        返回:
        main_category: 主类别名称
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

    def visualize_stereo_selection(self, results, output_dir):
        """
        可视化立体声音频的声道选择结果

        参数:
        results: 处理结果列表，包含左右声道V值
        output_dir: 输出目录
        """
        print("=== Starting stereo channel selection visualization ===")

        # Define color schemes for 5 main categories
        main_category_colors = {
            'High Ecology': '#FF6B6B',  # Red series
            'Low Ecology': '#DDA0DD',  # Purple series
            'speech': '#45B7D1',  # Blue series
            'music': '#96CEB4',  # Green series
            'Unknown Source': '#FFEAA7',  # Yellow series
        }

        # 解析每个音频文件的选择结果
        selection_data = []
        for result in results:
            filename = result['audio_file']
            left_variation = result['left_variation_sum']
            right_variation = result['right_variation_sum']

            # 解析文件名获取左右声道类别
            left_category, right_category = self.parse_stereo_filename(filename)

            if left_category is None or right_category is None:
                print(f"Skipping file {filename}: Unable to parse category information")
                continue

            # Determine which side to select (higher V value)
            if left_variation > right_variation:
                selected_category = left_category
                selected_side = 'Left'
                v_value = left_variation
                other_v_value = right_variation
            else:
                selected_category = right_category
                selected_side = 'Right'
                v_value = right_variation
                other_v_value = left_variation

            # Get main category
            main_category = self.get_main_category(selected_category)

            # Skip file if main category is Unknown
            if main_category == 'Unknown':
                print(f"Skipping file {filename}: Unable to identify main category")
                continue

            selection_data.append({
                'filename': filename,
                'left_category': left_category,
                'right_category': right_category,
                'left_variation': left_variation,
                'right_variation': right_variation,
                'selected_category': selected_category,
                'selected_side': selected_side,
                'v_value': v_value,
                'other_v_value': other_v_value,
                'main_category': main_category,
                'v_difference': abs(left_variation - right_variation)
            })

        if not selection_data:
            print("No valid selection data")
            return

        # Convert to DataFrame
        df = pd.DataFrame(selection_data)

        # Statistics by main category
        main_category_stats = df.groupby('main_category').agg({
            'filename': 'count',
            'v_value': ['mean', 'std'],
            'v_difference': 'mean'
        }).round(4)

        print("\nMain category selection statistics:")
        print(main_category_stats)

        # Statistics by sub-category
        sub_category_stats = df.groupby('selected_category').agg({
            'filename': 'count',
            'v_value': ['mean', 'std'],
            'v_difference': 'mean'
        }).round(4)

        print("\nSub-category selection statistics:")
        print(sub_category_stats)

        # Plot visualization charts
        self.plot_stereo_selection_charts(df, main_category_colors, output_dir)

        # Plot pairing and selection relationship charts
        self.plot_pairing_and_selection_analysis(df, main_category_colors, output_dir)

        # Plot pairing selection pattern analysis
        self.plot_pairing_selection_patterns(df, main_category_colors, output_dir)

        # Save results
        self.save_stereo_selection_results(df, output_dir)

        print(f"\nStereo selection visualization completed, results saved to: {output_dir}")

    def plot_stereo_selection_charts(self, df, main_category_colors, output_dir):
        """
        绘制立体声选择的可视化图表

        参数:
        df: 选择数据DataFrame
        main_category_colors: 主类别颜色映射
        output_dir: 输出目录
        """
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')

        # 1. Main category selection distribution pie chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stereo Audio Channel Selection Analysis', fontsize=16, fontweight='bold')

        # Main category selection distribution
        main_cat_counts = df['main_category'].value_counts()
        colors_main = [main_category_colors.get(cat, '#DDA0DD') for cat in main_cat_counts.index]

        axes[0, 0].pie(main_cat_counts.values, labels=main_cat_counts.index, autopct='%1.1f%%',
                       colors=colors_main, startangle=90)
        axes[0, 0].set_title('Main Category Selection Distribution', fontsize=14, fontweight='bold')

        # Sub-category selection distribution (top 10)
        sub_cat_counts = df['selected_category'].value_counts().head(10)
        axes[0, 1].bar(range(len(sub_cat_counts)), sub_cat_counts.values,
                       color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Sub-category Selection Distribution (Top 10)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Sub-category')
        axes[0, 1].set_ylabel('Selection Count')
        axes[0, 1].set_xticks(range(len(sub_cat_counts)))
        axes[0, 1].set_xticklabels(sub_cat_counts.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # V-value distribution histogram
        axes[0, 2].hist(df['v_value'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Selected V-value Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('V-value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)

        # Left vs Right channel V-value comparison scatter plot
        # 过滤掉main_category为NaN的行
        valid_df = df.dropna(subset=['main_category'])
        if not valid_df.empty:
            # 为每个点创建一半一半的颜色
            for _, row in valid_df.iterrows():
                left_cat = row['left_category']
                right_cat = row['right_category']

                # 获取左右声道的颜色
                left_color = main_category_colors.get(self.get_main_category(left_cat), '#DDA0DD')
                right_color = main_category_colors.get(self.get_main_category(right_cat), '#DDA0DD')

                # 创建一半一半的标记
                axes[1, 0].scatter(row['left_variation'], row['right_variation'],
                                   c=[left_color], alpha=0.8, s=100,
                                   edgecolors=right_color, linewidths=3)
        axes[1, 0].plot([df['left_variation'].min(), df['left_variation'].max()],
                        [df['left_variation'].min(), df['left_variation'].max()],
                        'r--', alpha=0.5, label='y=x')
        axes[1, 0].set_title('Left vs Right Channel V-value Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Left Channel V-value')
        axes[1, 0].set_ylabel('Right Channel V-value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Main category V-value box plot
        # 过滤掉main_category为NaN的行
        valid_df = df.dropna(subset=['main_category'])
        if not valid_df.empty:
            main_categories = valid_df['main_category'].unique()
            box_data = [valid_df[valid_df['main_category'] == cat]['v_value'].values for cat in main_categories]
            box_colors = [main_category_colors.get(cat, '#DDA0DD') for cat in main_categories]
        else:
            main_categories = []
            box_data = []
            box_colors = []

        bp = axes[1, 1].boxplot(box_data, patch_artist=True, labels=main_categories)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        axes[1, 1].set_title('Main Category V-value Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('V-value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # V-value difference distribution
        axes[1, 2].hist(df['v_difference'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 2].set_title('Left-Right V-value Difference Distribution', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('V-value Difference')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save main chart
        main_chart_path = os.path.join(output_dir, "stereo_selection_analysis.png")
        plt.savefig(main_chart_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Plot sub-category selection distribution by main category
        self.plot_subcategory_selection_by_main(df, main_category_colors, output_dir)

        # 3. Plot selection decision tree

        print(f"Main chart saved: {main_chart_path}")

    def plot_subcategory_selection_by_main(self, df, main_category_colors, output_dir):
        """
        Plot sub-category selection distribution by main category

        Parameters:
        df: Selection data DataFrame
        main_category_colors: Main category color mapping
        output_dir: Output directory
        """
        main_categories = df['main_category'].unique()

        for main_cat in main_categories:
            if main_cat == 'Unknown':
                continue

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Get data for this main category
            main_cat_data = df[df['main_category'] == main_cat]
            sub_cat_counts = main_cat_data['selected_category'].value_counts()

            # Generate colors with same hue but different brightness
            base_color = main_category_colors.get(main_cat, '#DDA0DD')
            base_rgb = plt.matplotlib.colors.to_rgb(base_color)

            # Create gradient colors
            n_subcats = len(sub_cat_counts)
            colors = []
            for i in range(n_subcats):
                # Adjust brightness from dark to light
                brightness = 0.3 + 0.7 * (i / (n_subcats - 1)) if n_subcats > 1 else 0.6
                color = tuple(c * brightness for c in base_rgb)
                colors.append(color)

            # Draw bar chart
            bars = ax.bar(range(len(sub_cat_counts)), sub_cat_counts.values,
                          color=colors, alpha=0.8, edgecolor='black')

            ax.set_title(f'{main_cat} - Sub-category Selection Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sub-category')
            ax.set_ylabel('Selection Count')
            ax.set_xticks(range(len(sub_cat_counts)))
            ax.set_xticklabels(sub_cat_counts.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()

            # Save chart
            safe_main_cat_name = main_cat.replace(' ', '_')
            chart_path = os.path.join(output_dir, f"subcategory_selection_{safe_main_cat_name}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  {main_cat} sub-category selection chart saved: {chart_path}")

    def plot_selection_decision_tree(self, df, main_category_colors, output_dir):
        """
        Plot selection decision visualization tree

        Parameters:
        df: Selection data DataFrame
        main_category_colors: Main category color mapping
        output_dir: Output directory
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Count left and right channel selections
        left_selected = len(df[df['selected_side'] == 'Left'])
        right_selected = len(df[df['selected_side'] == 'Right'])

        # Create decision tree visualization
        y_positions = [0.8, 0.6, 0.4, 0.2]

        # Root node
        ax.text(0.5, y_positions[0], 'Stereo Audio Files', ha='center', va='center',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Left and right channel branches
        ax.plot([0.5, 0.25], [y_positions[0], y_positions[1]], 'k-', linewidth=2)
        ax.plot([0.5, 0.75], [y_positions[0], y_positions[1]], 'k-', linewidth=2)

        ax.text(0.25, y_positions[1], f'Left Channel\n({left_selected} files)', ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

        ax.text(0.75, y_positions[1], f'Right Channel\n({right_selected} files)', ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Show selection results by main category
        main_cat_counts = df['main_category'].value_counts()
        x_positions = np.linspace(0.1, 0.9, len(main_cat_counts))

        for i, (main_cat, count) in enumerate(main_cat_counts.items()):
            color = main_category_colors.get(main_cat, '#DDA0DD')

            # Draw main category node
            ax.text(x_positions[i], y_positions[2], f'{main_cat}\n({count} files)',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

            # Draw connection lines
            if left_selected > right_selected:
                ax.plot([0.25, x_positions[i]], [y_positions[1], y_positions[2]],
                        color=color, alpha=0.5, linewidth=1)
            else:
                ax.plot([0.75, x_positions[i]], [y_positions[1], y_positions[2]],
                        color=color, alpha=0.5, linewidth=1)

        # Add statistics
        stats_text = f"""
        Total Files: {len(df)}
        Left Channel Selected: {left_selected} ({left_selected / len(df) * 100:.1f}%)
        Right Channel Selected: {right_selected} ({right_selected / len(df) * 100:.1f}%)

        Average V-value Difference: {df['v_difference'].mean():.4f}
        Max V-value Difference: {df['v_difference'].max():.4f}
        """

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.set_title('Stereo Audio Channel Selection Decision Tree', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save decision tree chart
        tree_path = os.path.join(output_dir, "selection_decision_tree.png")
        plt.savefig(tree_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Selection decision tree saved: {tree_path}")

    def plot_pairing_and_selection_analysis(self, df, main_category_colors, output_dir):
        """
        Plot pairing and selection analysis for stereo audio files

        Parameters:
        df: Selection data DataFrame
        main_category_colors: Main category color mapping
        output_dir: Output directory
        """
        print("=== Creating Pairing and Selection Analysis ===")

        # 1. 配对频率热力图
        self.plot_pairing_heatmap(df, main_category_colors, output_dir)

        # 2. 配对选择决策图
        self.plot_pairing_selection_decisions(df, main_category_colors, output_dir)

        # 4. 选择倾向分析
        self.plot_selection_tendency_analysis(df, main_category_colors, output_dir)

    def plot_pairing_heatmap(self, df, main_category_colors, output_dir):
        """绘制小类别配对选择决策热力图"""
        # 定义标准类别顺序
        target_categories = [
            'Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',  # 高生态效度
            'Computer keyboard', 'Helicopter', 'Chicken, rooster',  # 低生态效度
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # 未知声源
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

        ax.set_title('Category Pairing Selection Results\n(Color = Selected Category)',
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
        heatmap_path = os.path.join(output_dir, "category_pairing_selection_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Pairing selection heatmap saved: {heatmap_path}")

    def plot_pairing_selection_decisions(self, df, main_category_colors, output_dir):
        """Plot pairing selection decision charts"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Pairing and Selection Decision Analysis', fontsize=18, fontweight='bold')

        # 1. Pairing selection decision scatter plot
        ax1 = axes[0, 0]

        # Create colors for each pairing
        pair_colors = {}
        color_idx = 0
        for _, row in df.iterrows():
            pair = f"{row['left_category']} + {row['right_category']}"
            if pair not in pair_colors:
                pair_colors[pair] = plt.cm.tab20(color_idx % 20)
                color_idx += 1

        # Draw scatter plot
        for _, row in df.iterrows():
            pair = f"{row['left_category']} + {row['right_category']}"
            color = pair_colors[pair]

            # Mark shape based on selection result
            marker = 'o' if row['selected_side'] == 'Left' else 's'
            size = 100 + abs(row['v_difference']) * 10  # Size represents V-value difference

            ax1.scatter(row['left_variation'], row['right_variation'],
                        c=[color], marker=marker, s=size, alpha=0.7, edgecolors='black')

        ax1.plot([df['left_variation'].min(), df['left_variation'].max()],
                 [df['left_variation'].min(), df['left_variation'].max()],
                 'r--', alpha=0.5, label='Equal V-values')
        ax1.set_title('Pairing Selection Decisions\n(Circle=Left Selected, Square=Right Selected)',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Left Channel V-value')
        ax1.set_ylabel('Right Channel V-value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Pairing selection statistics bar chart
        ax2 = axes[0, 1]

        # Count selection results for each pairing
        pair_stats = {}
        for _, row in df.iterrows():
            pair = f"{row['left_category']} + {row['right_category']}"
            if pair not in pair_stats:
                pair_stats[pair] = {'left_selected': 0, 'right_selected': 0, 'total': 0}

            pair_stats[pair]['total'] += 1
            if row['selected_side'] == 'Left':
                pair_stats[pair]['left_selected'] += 1
            else:
                pair_stats[pair]['right_selected'] += 1

        # Draw stacked bar chart
        pairs = list(pair_stats.keys())
        left_counts = [pair_stats[pair]['left_selected'] for pair in pairs]
        right_counts = [pair_stats[pair]['right_selected'] for pair in pairs]

        x_pos = np.arange(len(pairs))
        ax2.bar(x_pos, left_counts, label='Left Selected', color='lightcoral', alpha=0.8)
        ax2.bar(x_pos, right_counts, bottom=left_counts, label='Right Selected',
                color='lightgreen', alpha=0.8)

        ax2.set_title('Selection Results by Pairing', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Category Pairs')
        ax2.set_ylabel('Selection Count')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(pairs, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. V-value difference distribution (grouped by pairing)
        ax3 = axes[1, 0]

        pair_differences = {}
        for _, row in df.iterrows():
            pair = f"{row['left_category']} + {row['right_category']}"
            if pair not in pair_differences:
                pair_differences[pair] = []
            pair_differences[pair].append(row['v_difference'])

        # Draw box plot
        pair_names = list(pair_differences.keys())
        diff_data = [pair_differences[pair] for pair in pair_names]

        bp = ax3.boxplot(diff_data, patch_artist=True, labels=pair_names)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(plt.cm.tab20(i % 20))

        ax3.set_title('V-value Difference Distribution by Pairing', fontsize=14, fontweight='bold')
        ax3.set_ylabel('V-value Difference')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Selection tendency analysis
        ax4 = axes[1, 1]

        # Calculate selection tendency for each sub-category
        category_tendency = {}
        for _, row in df.iterrows():
            left_cat = row['left_category']
            right_cat = row['right_category']

            if left_cat not in category_tendency:
                category_tendency[left_cat] = {'as_left': 0, 'as_right': 0, 'total_left': 0, 'total_right': 0}
            if right_cat not in category_tendency:
                category_tendency[right_cat] = {'as_left': 0, 'as_right': 0, 'total_left': 0, 'total_right': 0}

            category_tendency[left_cat]['total_left'] += 1
            category_tendency[right_cat]['total_right'] += 1

            if row['selected_side'] == 'Left':
                category_tendency[left_cat]['as_left'] += 1
            else:
                category_tendency[right_cat]['as_right'] += 1

        # Calculate selection rates
        categories = list(category_tendency.keys())
        left_rates = []
        right_rates = []

        for cat in categories:
            left_rate = category_tendency[cat]['as_left'] / category_tendency[cat]['total_left'] if \
                category_tendency[cat]['total_left'] > 0 else 0
            right_rate = category_tendency[cat]['as_right'] / category_tendency[cat]['total_right'] if \
                category_tendency[cat]['total_right'] > 0 else 0
            left_rates.append(left_rate)
            right_rates.append(right_rate)

        x_pos = np.arange(len(categories))
        width = 0.35

        ax4.bar(x_pos - width / 2, left_rates, width, label='When as Left Channel',
                color='lightcoral', alpha=0.8)
        ax4.bar(x_pos + width / 2, right_rates, width, label='When as Right Channel',
                color='lightgreen', alpha=0.8)

        ax4.set_title('Selection Rate by Category Position', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Categories')
        ax4.set_ylabel('Selection Rate')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        decision_path = os.path.join(output_dir, "pairing_selection_decisions.png")
        plt.savefig(decision_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Pairing selection decisions saved: {decision_path}")

    def plot_selection_tendency_analysis(self, df, main_category_colors, output_dir):
        """Plot selection tendency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Selection Tendency Analysis', fontsize=16, fontweight='bold')

        # 1. Main category selection tendency
        ax1 = axes[0, 0]

        main_cat_tendency = {}
        for _, row in df.iterrows():
            main_cat = row['main_category']
            if main_cat not in main_cat_tendency:
                main_cat_tendency[main_cat] = {'left': 0, 'right': 0, 'total': 0}

            main_cat_tendency[main_cat]['total'] += 1
            if row['selected_side'] == 'Left':
                main_cat_tendency[main_cat]['left'] += 1
            else:
                main_cat_tendency[main_cat]['right'] += 1

        main_cats = list(main_cat_tendency.keys())
        left_rates = [main_cat_tendency[cat]['left'] / main_cat_tendency[cat]['total'] for cat in main_cats]
        right_rates = [main_cat_tendency[cat]['right'] / main_cat_tendency[cat]['total'] for cat in main_cats]

        x_pos = np.arange(len(main_cats))
        width = 0.35

        ax1.bar(x_pos - width / 2, left_rates, width, label='Left Channel Selected',
                color='lightcoral', alpha=0.8)
        ax1.bar(x_pos + width / 2, right_rates, width, label='Right Channel Selected',
                color='lightgreen', alpha=0.8)

        ax1.set_title('Selection Rate by Main Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Main Categories')
        ax1.set_ylabel('Selection Rate')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(main_cats, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. V-value difference and selection relationship
        ax2 = axes[0, 1]

        left_selected = df[df['selected_side'] == 'Left']['v_difference']
        right_selected = df[df['selected_side'] == 'Right']['v_difference']

        ax2.hist(left_selected, bins=20, alpha=0.7, label='Left Selected',
                 color='lightcoral', edgecolor='black')
        ax2.hist(right_selected, bins=20, alpha=0.7, label='Right Selected',
                 color='lightgreen', edgecolor='black')

        ax2.set_title('V-value Difference Distribution by Selection', fontsize=14, fontweight='bold')
        ax2.set_xlabel('V-value Difference')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Selection decision certainty analysis
        ax3 = axes[1, 0]

        # Calculate selection certainty for each file (absolute value of V-value difference)
        certainty = df['v_difference'].abs()

        ax3.hist(certainty, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(certainty.mean(), color='red', linestyle='--',
                    label=f'Mean: {certainty.mean():.2f}')
        ax3.axvline(certainty.median(), color='orange', linestyle='--',
                    label=f'Median: {certainty.median():.2f}')

        ax3.set_title('Selection Decision Certainty\n(Absolute V-value Difference)',
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('Absolute V-value Difference')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Pairing pattern analysis
        ax4 = axes[1, 1]

        # Count most common pairing patterns
        pair_counts = {}
        for _, row in df.iterrows():
            pair = tuple(sorted([row['left_category'], row['right_category']]))
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += 1

        # Take top 10 most common pairings
        top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        pair_names = [f"{pair[0]} + {pair[1]}" for pair, _ in top_pairs]
        counts = [count for _, count in top_pairs]

        bars = ax4.bar(range(len(pair_names)), counts, color='lightblue', alpha=0.8, edgecolor='black')
        ax4.set_title('Most Common Category Pairings', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Category Pairs')
        ax4.set_ylabel('Frequency')
        ax4.set_xticks(range(len(pair_names)))
        ax4.set_xticklabels(pair_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save chart
        tendency_path = os.path.join(output_dir, "selection_tendency_analysis.png")
        plt.savefig(tendency_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Selection tendency analysis saved: {tendency_path}")

    def save_stereo_selection_results(self, df, output_dir):
        """
        Save stereo selection results

        Parameters:
        df: Selection data DataFrame
        output_dir: Output directory
        """
        # Save detailed results
        detailed_path = os.path.join(output_dir, "stereo_selection_detailed.csv")
        df.to_csv(detailed_path, index=False, encoding='utf-8-sig')

        # Save summary statistics
        summary_stats = []

        # Statistics by main category
        main_cat_stats = df.groupby('main_category').agg({
            'filename': 'count',
            'v_value': ['mean', 'std', 'min', 'max'],
            'v_difference': 'mean'
        }).round(4)

        # Statistics by sub-category
        sub_cat_stats = df.groupby('selected_category').agg({
            'filename': 'count',
            'v_value': ['mean', 'std', 'min', 'max'],
            'v_difference': 'mean'
        }).round(4)

        # Left/Right channel selection statistics
        side_stats = df['selected_side'].value_counts()

        # Save statistics results
        with open(os.path.join(output_dir, "stereo_selection_summary.txt"), 'w', encoding='utf-8') as f:
            f.write("Stereo Audio Channel Selection Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. Overall Statistics\n")
            f.write(f"Total Files: {len(df)}\n")
            f.write(
                f"Left Channel Selected: {side_stats.get('Left', 0)} ({side_stats.get('Left', 0) / len(df) * 100:.1f}%)\n")
            f.write(
                f"Right Channel Selected: {side_stats.get('Right', 0)} ({side_stats.get('Right', 0) / len(df) * 100:.1f}%)\n\n")

            f.write("2. Main Category Selection Statistics\n")
            f.write(str(main_cat_stats) + "\n\n")

            f.write("3. Sub-category Selection Statistics\n")
            f.write(str(sub_cat_stats) + "\n\n")

            f.write("4. V-value Statistics\n")
            f.write(f"Average V-value: {df['v_value'].mean():.4f}\n")
            f.write(f"V-value Std Dev: {df['v_value'].std():.4f}\n")
            f.write(f"Average V-value Difference: {df['v_difference'].mean():.4f}\n")
            f.write(f"Max V-value Difference: {df['v_difference'].max():.4f}\n")

        print(f"Selection results saved:")
        print(f"  Detailed results: {detailed_path}")
        print(f"  Summary statistics: {output_dir}/stereo_selection_summary.txt")

    def save_category_variations(self, category_variations, all_variations, output_dir):
        """
        保存类别帧变化和结果

        参数:
        category_variations: 按类别组织的帧变化数据
        all_variations: 所有音频的帧变化数据
        output_dir: 输出目录
        """
        # 保存详细结果
        detailed_data = []
        for var in all_variations:
            detailed_data.append({
                'main_category': var['main_category'],
                'sub_category': var['sub_category'],
                'file_name': var['file_name'],
                'variation_sum': var['variation_sum']
            })

        df_detailed = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(output_dir, "category_frame_variations_detailed.csv")
        df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')

        # 保存汇总统计
        summary_data = []
        for main_cat, sub_cats_data in category_variations.items():
            for sub_cat, data in sub_cats_data.items():
                if data:
                    variation_sums = [d['variation_sum'] for d in data]

                    summary_data.append({
                        'main_category': main_cat,
                        'sub_category': sub_cat,
                        'file_count': len(data),
                        'variation_mean': np.mean(variation_sums),
                        'variation_std': np.std(variation_sums),
                        'variation_min': np.min(variation_sums),
                        'variation_max': np.max(variation_sums)
                    })

        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "category_frame_variations_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

        # 保存numpy数据
        np_path = os.path.join(output_dir, "category_variations_data.npz")
        np.savez(np_path,
                 category_variations=category_variations,
                 all_variations=all_variations,
                 detailed_data=df_detailed.values,
                 summary_data=df_summary.values,
                 column_names_detailed=df_detailed.columns.tolist(),
                 column_names_summary=df_summary.columns.tolist())

        print(f"结果已保存:")
        print(f"  详细结果: {detailed_path}")
        print(f"  汇总统计: {summary_path}")
        print(f"  numpy数据: {np_path}")


def main():
    """Main function"""
    print("=== Whisper Stereo Feature Extraction Tool ===")

    # Set parameters
    model_size = "base"  # Options: "tiny", "base", "small", "medium", "large"
    frame_duration_ms = 8  # 8ms per frame

    # Set paths
    # input_dir = r"D:\D\research\audioset下载\classified_audio"  # Original audio directory
    input_dir = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"  # Original audio directory
    output_dir = r"D:\D\research\audioset下载\test120"  # Output directory
    features_data_path = r"D:\D\research\audioset下载\特征"  # Feature data path

    # Create feature extractor
    extractor = WhisperStereoFeatureExtractor(model_size=model_size)

    # Batch process stereo audio files
    results = extractor.process_audio_pair(
        input_path=input_dir,
        out_path=output_dir,
        max_files=1000)  # Set a large number to ensure processing all files

    # Visualize stereo channel selection results
    if results:
        extractor.visualize_stereo_selection(results, output_dir)

    # Calculate frame variation sum distribution for each category (commented out)
    # extractor.calculate_category_frame_variations(input_dir, output_dir)


if __name__ == "__main__":
    main()
