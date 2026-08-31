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

warnings.filterwarnings("ignore")


class WhisperStereoFeatureExtractor:
    """使用Whisper提取音频特征的类（支持单声道和立体声）"""

    def __init__(self, model_size="base", device=None):
        """
        初始化Whisper模型

        参数:
        model_size: Whisper模型大小 ("tiny", "base", "small", "medium", "large")
        device: 计算设备 ("cpu", "cuda", "auto")
        """
        self.model_size = model_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在加载Whisper模型: {model_size}")
        print(f"使用设备: {self.device}")

        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("Whisper模型加载成功")
        except Exception as e:
            print(f"Whisper模型加载失败: {e}")
            raise

    def extract_whisper_features(self, audio_data, frame_duration_ms=8):
        """
        使用Whisper提取整段音频的编码器特征，并按Whisper原生时间步(~10ms)返回。

        参数:
        audio_data: 一维波形(np.float32, 16kHz)
        frame_duration_ms: 保留参数(忽略)，时间步固定为~10ms

        返回:
        features: (n_steps, feature_dim) 的特征矩阵
        timestamps: 每个时间步的时间戳(秒)，步长约0.01s
        """
        try:
            # 确保数据类型与形状
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.detach().cpu().numpy().astype(np.float32)
            else:
                audio_np = np.asarray(audio_data, dtype=np.float32)
            # 强制为一维 (samples,)
            audio_np = np.squeeze(audio_np)
            if audio_np.ndim != 1:
                # 如果仍非一维，尝试取第一维为时间轴
                audio_np = np.reshape(audio_np, (-1,))
            if audio_np.size == 0:
                raise ValueError("empty audio after squeeze")

            # 计算整段mel: (80, n_frames)
            with torch.no_grad():
                # 调试与健壮性：确保一维、连续、float32
                audio_np = np.ascontiguousarray(audio_np, dtype=np.float32)
                # 直接以torch张量调用
                audio_t = torch.from_numpy(audio_np)
                if audio_t.ndim != 1:
                    raise ValueError(f"audio must be 1D, got shape {tuple(audio_t.shape)}")
                # 记录形状与dtype
                # print(f"[DEBUG] audio shape: {tuple(audio_t.shape)}, dtype: {audio_t.dtype}")
                mel = whisper_log_mel_spectrogram(audio_t)  # (80, n_frames)
                # pad/trim 到Whisper要求的3000帧
                mel = whisper_pad_or_trim(mel, WHISPER_N_FRAMES)  # (80, 3000)
                # 加batch维度: (1, 80, 3000)
                mel = mel.unsqueeze(0).to(self.device)

                # 过编码器: (1, n_frames, d_model)
                enc = self.model.encoder(mel)
                features = enc.squeeze(0).detach().cpu().numpy()  # (n_frames, d_model)

            # Whisper时间步约为10ms
            n_steps = features.shape[0]
            timestamps = (np.arange(n_steps, dtype=np.float32) * 0.01).tolist()

            return features, timestamps
        except Exception as e:
            print(f"提取Whisper特征失败: {e}")
            return None, None

    def extract_librosa_features(self, audio_data, frame_duration_ms=8):
        """
        使用librosa提取MFCC特征作为备选方案

        参数:
        audio_data: 音频数据
        frame_duration_ms: 帧长度（毫秒）

        返回:
        features: 提取的特征矩阵 (n_frames, feature_dim)
        timestamps: 时间戳列表
        """
        try:
            sr = 16000
            frame_length = int(sr * frame_duration_ms / 1000)
            total_frames = len(audio_data) // frame_length

            if total_frames == 0:
                print("警告: 音频太短，无法提取特征")
                return None, None

            features_list = []
            timestamps = []

            print(f"使用librosa提取MFCC特征，总帧数: {total_frames}")

            # 逐帧提取MFCC特征
            for i in tqdm(range(total_frames), desc="提取librosa特征"):
                start_idx = i * frame_length
                end_idx = start_idx + frame_length
                frame_audio = audio_data[start_idx:end_idx]

                # 确保帧长度正确
                if len(frame_audio) < frame_length:
                    frame_audio = np.pad(frame_audio, (0, frame_length - len(frame_audio)), 'constant')

                # 提取MFCC特征
                mfcc = librosa.feature.mfcc(y=frame_audio, sr=sr, n_mfcc=13)
                frame_features = np.mean(mfcc, axis=1)  # 对时间维度取平均

                features_list.append(frame_features)
                timestamps.append(i * frame_duration_ms / 1000)

            return np.array(features_list), timestamps

        except Exception as e:
            print(f"提取librosa特征失败: {e}")
            return None, None

    def calculate_feature_variation_sum(self, left_features, right_features):
        """
        计算左右声道所有帧的表征变化之和

        参数:
        left_features: 左声道特征矩阵 (n_frames, feature_dim)
        right_features: 右声道特征矩阵 (n_frames, feature_dim)

        返回:
        left_variation_sum: 左声道帧变化和
        right_variation_sum: 右声道帧变化和
        left_minus_right_diff: 左声道变化和减去右声道变化和的差值
        variation_details: 每帧的变化详情
        """
        try:
            # 确保两个特征矩阵形状一致
            min_frames = min(len(left_features), len(right_features))
            left_features = left_features[:min_frames]
            right_features = right_features[:min_frames]

            # 计算左声道每帧的变化（与前一帧的L2距离）
            left_frame_variations = []
            for i in range(min_frames):
                if i == 0:
                    # 第一帧，变化为0
                    left_frame_variations.append(0.0)
                else:
                    # 计算与前一帧的L2距离
                    left_variation = np.linalg.norm(left_features[i] - left_features[i - 1])
                    left_frame_variations.append(left_variation)

            # 计算右声道每帧的变化（与前一帧的L2距离）
            right_frame_variations = []
            for i in range(min_frames):
                if i == 0:
                    # 第一帧，变化为0
                    right_frame_variations.append(0.0)
                else:
                    # 计算与前一帧的L2距离
                    right_variation = np.linalg.norm(right_features[i] - right_features[i - 1])
                    right_frame_variations.append(right_variation)

            # 计算左右声道帧变化和
            left_variation_sum = np.sum(left_frame_variations)
            right_variation_sum = np.sum(right_frame_variations)

            # 计算左减右的差值
            left_minus_right_diff = left_variation_sum - right_variation_sum

            # 计算统计信息
            variation_stats = {
                'left_variation_sum': left_variation_sum,
                'right_variation_sum': right_variation_sum,
                'left_minus_right_diff': left_minus_right_diff,
                'mean_left_variation': np.mean(left_frame_variations[1:]),  # 排除第一帧
                'mean_right_variation': np.mean(right_frame_variations[1:]),  # 排除第一帧
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
            print(f"计算表征变化失败: {e}")
            return None, None, None, None, None, None

    def process_directory(self, input_dir, output_dir, frame_duration_ms=8, file_pattern="*.wav"):
        """
        批量处理目录中的音频文件（只处理指定的类别）

        参数:
        input_dir: 输入目录
        output_dir: 输出目录
        frame_duration_ms: 帧长度（毫秒）
        file_pattern: 文件匹配模式
        """
        print(f"开始批量处理目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"帧长度: {frame_duration_ms}ms")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 定义要处理的特定类别
        target_categories = [
            'Alarm', 'Crying, sobbing', 'Roar',  # 高生态效度
            'Computer keyboard', 'Fireworks', 'Chicken, rooster',  # 低生态效度
            'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',  # speech
            'Bass drum', 'Funny music', 'Sad music',  # music
            'Pulse', 'Whack, thwack', 'Crumpling, crinkling'  # 未知声源
        ]

        print(f"目标类别: {target_categories}")

        # 按类别组织文件
        category_files = {}
        total_files = 0

        for category in target_categories:
            category_path = Path(input_dir) / category
            if category_path.exists() and category_path.is_dir():
                # 查找该类别目录下的音频文件
                audio_files = list(category_path.glob(file_pattern))
                if audio_files:
                    category_files[category] = audio_files
                    total_files += len(audio_files)
                    print(f"  {category}: {len(audio_files)} 个文件")
                else:
                    print(f"  {category}: 未找到音频文件")
            else:
                print(f"  {category}: 目录不存在")

        if total_files == 0:
            print(f"在指定类别中未找到匹配的音频文件")
            return

        print(f"总共找到 {total_files} 个音频文件")

        # 处理结果列表
        all_results = []

        # 按类别逐个处理音频文件
        for category, files in category_files.items():
            print(f"\n处理类别: {category}")
            for audio_file in tqdm(files, desc=f"处理{category}"):
                result = self.process_audio_file(str(audio_file), frame_duration_ms)
                if result is not None:
                    # 添加类别信息到结果中
                    result['category'] = category
                    all_results.append(result)

        # 保存结果
        self.save_results(all_results, output_dir)

        print(f"\n批量处理完成，共处理 {len(all_results)} 个文件")

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

        print(f"结果已保存:")
        print(f"  汇总结果: {csv_path}")
        print(f"  numpy数据: {np_path}")
        print(f"  直方图: {output_dir}/variation_histograms.png")

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
            '高生态效度': ['Alarm', 'Crying, sobbing', 'Roar'],
            '低生态效度': ['Computer keyboard', 'Fireworks', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music'],
            '未知声源': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
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
    """主函数"""
    print("=== Whisper立体声特征提取工具 ===")

    # 设置参数
    model_size = "base"  # 可选: "tiny", "base", "small", "medium", "large"
    frame_duration_ms = 8  # 8ms一帧

    # 设置路径
    input_dir = r"D:\D\research\audioset下载\classified_audio"  # 原始音频目录
    output_dir = r"D:\D\research\audioset下载\whisper_stereo_features"  # 输出目录

    # 创建特征提取器
    extractor = WhisperStereoFeatureExtractor(model_size=model_size)

    # 批量处理立体声音频
    extractor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        frame_duration_ms=frame_duration_ms,
        file_pattern="*.wav"
    )

    # 计算每个类别的帧变化和分布 (已注释)
    extractor.calculate_category_frame_variations(input_dir, output_dir)


if __name__ == "__main__":
    main()
