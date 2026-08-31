#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用individual_choice_with_position.py中的所有注意力模型测试hanglei数据
- 加载hanglei的三个session数据
- 筛选一致的trials
- 现场提取音频特征
- 使用所有注意力模型进行分类
- 输出结果对比
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要的函数
from spatiotemporal_glm_model import (
    parse_experiment_data_txt,
    parse_pair_name
)
from linear_regression_selected_features import (
    build_consistent_folds,
)
from linear_regression_loudness_models import (
    DEFAULT_AUDIO_BASE_DIR,
    PAIR_FEATURE_CACHE,
    STD_CACHE,
    _cache_stats
)

# 导入feature_stage_attention_model中的模型和函数
from feature_stage_attention_model import (
    FeatureWeightedModel,
    FeatureStageAttentionModel,
    SharedEarComparisonModel,
    IndependentEarComparisonModel,
    StageAwareSharedComparisonModel,
    StageAwareSharedComparisonUnified,
    StageFeatureDataset,
    DEFAULT_STAGE_SECONDS,
    RNG_SEED,
    DEVICE,
    get_feature_configs_from_linear_regression,
    SR,
    HOP
)

# 导入attention_loudness_model中的FeatureTimeSeriesDataset
from attention_loudness_model import FeatureTimeSeriesDataset

# 导入individual_choice_with_position中的函数
from individual_choice_with_position import (
    check_left_right_consistency,
    extract_features_for_attention_model,
    fit_attention_choice_model,
    reset_feature_caches,
    ATTENTION_MODEL_CONFIGS,
    plot_attention_model_comparison,
    save_attention_results,
    _parse_pair_tokens
)

# 导入eeg_audio_fusion_classify_hanglei中的函数
from eeg_audio_fusion_classify_hanglei import (
    load_hanglei_trials,
    load_all_mappings,
    MAPPING_FILES,
    BENCHMARK_FILES,
    AUDIO_BASE_DIR
)

CV_SEED = 42
N_FOLDS = 5

# 输出目录
OUTPUT_DIR = os.path.join(os.getcwd(), 'individual_choice_hanglei_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_stage_dataset_for_trials(consistent_trials, base_dir):
    """
    为consistent_trials构建StageFeatureDataset（现场提取）
    """
    print("\n=== 构建StageFeatureDataset（现场提取） ===")
    
    # 获取特征配置
    feature_configs = get_feature_configs_from_linear_regression()
    print(f"使用 {len(feature_configs)} 个特征配置")
    
    # 提取所有consistent_trials的pair_name
    pair_names_to_extract = [trial['pair_name'] for trial in consistent_trials]
    print(f"需要提取特征的音频对数量: {len(pair_names_to_extract)}")
    
    # 构建FeatureTimeSeriesDataset来提取特征
    dummy_targets = np.zeros(len(pair_names_to_extract), dtype=np.float32)
    print("正在构建FeatureTimeSeriesDataset并提取特征...")
    base_dataset = FeatureTimeSeriesDataset(
        pair_names_to_extract,
        base_dir,
        dummy_targets,
        feature_configs,
        sr=SR,
        hop_length=HOP,
        window_size=5,
        use_high_res=False,
    )
    
    if len(base_dataset) == 0:
        raise RuntimeError("无法构建FeatureTimeSeriesDataset，可能音频文件不存在")
    
    print(f"成功构建FeatureTimeSeriesDataset，包含 {len(base_dataset)} 个样本")
    
    # 构建StageFeatureDataset来计算全局特征
    stage_dataset = StageFeatureDataset(
        base_dataset,
        stage_seconds=DEFAULT_STAGE_SECONDS,
        feature_configs=feature_configs,
    )
    stage_dataset.pair_names = pair_names_to_extract
    
    print(f"阶段特征数据集大小: {len(stage_dataset)}")
    print(f"  阶段数量: {stage_dataset.n_stages}")
    print(f"  特征维度: {stage_dataset.feat_dim}")
    
    return stage_dataset


def build_balanced_trials(consistent_trials, pair_names):
    """
    构建平衡的标签数据集
    对于每个一致的trial，我们知道选择了哪个音频（A或B）
    通过选择保留AB版本还是BA版本来平衡标签，但不能改变实际选择
    
    策略：
    - 如果选择了A：可以用AB版本（choice=1）或BA版本（choice=2）
    - 如果选择了B：可以用AB版本（choice=2）或BA版本（choice=1）
    - 目标是让choice=1和choice=2的数量平衡
    """
    print(f"\n=== 构建平衡标签数据集 ===")
    print(f"  原始一致的trials数量: {len(consistent_trials)}")
    
    # 首先构建规范化键到AB和BA版本pair_name的映射
    normalized_to_pairs = {}
    
    # 按规范化键分组所有pair_names
    for pair_name in pair_names:
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        
        if key not in normalized_to_pairs:
            normalized_to_pairs[key] = {'ab': None, 'ba': None}
        
        # 判断是AB还是BA版本
        left_audio = f"{left_cat}_{left_id}"
        right_audio = f"{right_cat}_{right_id}"
        audio_a_str, audio_b_str = key
        
        if left_audio == audio_a_str and right_audio == audio_b_str:
            normalized_to_pairs[key]['ab'] = pair_name
        elif left_audio == audio_b_str and right_audio == audio_a_str:
            normalized_to_pairs[key]['ba'] = pair_name
    
    # 对于每个一致的trial，确定它选择了哪个音频，以及应该用AB还是BA版本
    trials_select_a = []  # 选择了A的trials
    trials_select_b = []   # 选择了B的trials
    
    for trial in consistent_trials:
        pair_name = trial['pair_name']
        parse_result = parse_pair_name(pair_name)
        if parse_result is None:
            continue
        
        left_cat, left_id, right_cat, right_id, _ = parse_result
        audio_a = f"{left_cat}_{left_id}"
        audio_b = f"{right_cat}_{right_id}"
        key = tuple(sorted([audio_a, audio_b]))
        
        if key not in normalized_to_pairs:
            continue
        
        ab_pair = normalized_to_pairs[key]['ab']
        ba_pair = normalized_to_pairs[key]['ba']
        
        if ab_pair is None or ba_pair is None:
            continue
        
        # 判断当前pair_name是AB还是BA版本
        current_left = f"{left_cat}_{left_id}"
        audio_a_str, audio_b_str = key
        is_ab_version = (current_left == audio_a_str)
        
        # 确定选择了哪个音频
        # 如果当前是AB版本：choice=1选A，choice=2选B
        # 如果当前是BA版本：choice=1选B，choice=2选A
        if is_ab_version:
            if trial['choice'] == 1:
                selected_audio = 'A'
            elif trial['choice'] == 2:
                selected_audio = 'B'
            else:
                continue
        else:  # BA版本
            if trial['choice'] == 1:
                selected_audio = 'B'
            elif trial['choice'] == 2:
                selected_audio = 'A'
            else:
                continue
        
        trial_info = {
            'original_trial': trial,
            'key': key,
            'ab_pair': ab_pair,
            'ba_pair': ba_pair,
            'selected_audio': selected_audio
        }
        
        if selected_audio == 'A':
            trials_select_a.append(trial_info)
        else:
            trials_select_b.append(trial_info)
    
    print(f"  选择A的trials数量: {len(trials_select_a)}")
    print(f"  选择B的trials数量: {len(trials_select_b)}")
    
    # 统计当前choice=1和choice=2的数量（如果都用AB版本）
    # 选择A的用AB版本→choice=1，选择B的用AB版本→choice=2
    current_choice1 = len(trials_select_a)
    current_choice2 = len(trials_select_b)
    
    print(f"  如果都用AB版本：choice=1有{current_choice1}个，choice=2有{current_choice2}个")
    
    # 目标：让choice=1和choice=2的数量平衡
    target_count = (len(trials_select_a) + len(trials_select_b)) // 2
    
    # 调整策略：
    # 如果choice=1太多，将一些选择A的trial改为BA版本（这样choice变成2）
    # 如果choice=2太多，将一些选择B的trial改为BA版本（这样choice变成1）
    
    balanced_trials = []
    
    if current_choice1 > current_choice2:
        # choice=1太多，需要将一些选择A的trial改为BA版本
        need_change = current_choice1 - target_count
        np.random.seed(CV_SEED)
        change_indices = np.random.choice(len(trials_select_a), need_change, replace=False)
        
        for i, trial_info in enumerate(trials_select_a):
            if i in change_indices:
                # 使用BA版本，选择A→choice=2（因为BA版本中A在右边）
                balanced_trials.append({
                    'pair_name': trial_info['ba_pair'],
                    'choice': 2,  # BA版本中选A（右边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
            else:
                # 使用AB版本，选择A→choice=1
                balanced_trials.append({
                    'pair_name': trial_info['ab_pair'],
                    'choice': 1,  # AB版本中选A（左边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
        
        # 选择B的trials都用AB版本，选择B→choice=2
        for trial_info in trials_select_b:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 2,  # AB版本中选B（右边）
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })
    
    elif current_choice2 > current_choice1:
        # choice=2太多，需要将一些选择B的trial改为BA版本
        need_change = current_choice2 - target_count
        np.random.seed(CV_SEED)
        change_indices = np.random.choice(len(trials_select_b), need_change, replace=False)
        
        # 选择A的trials都用AB版本，选择A→choice=1
        for trial_info in trials_select_a:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 1,  # AB版本中选A（左边）
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })
        
        for i, trial_info in enumerate(trials_select_b):
            if i in change_indices:
                # 使用BA版本，选择B→choice=1（因为BA版本中B在左边）
                balanced_trials.append({
                    'pair_name': trial_info['ba_pair'],
                    'choice': 1,  # BA版本中选B（左边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
            else:
                # 使用AB版本，选择B→choice=2
                balanced_trials.append({
                    'pair_name': trial_info['ab_pair'],
                    'choice': 2,  # AB版本中选B（右边）
                    'reaction_time': trial_info['original_trial'].get('reaction_time', None)
                })
    
    else:
        # 已经平衡，都用AB版本
        for trial_info in trials_select_a:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 1,
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })
        for trial_info in trials_select_b:
            balanced_trials.append({
                'pair_name': trial_info['ab_pair'],
                'choice': 2,
                'reaction_time': trial_info['original_trial'].get('reaction_time', None)
            })
    
    # 打乱顺序
    np.random.shuffle(balanced_trials)
    
    # 统计最终结果
    final_choice1 = sum(1 for t in balanced_trials if t['choice'] == 1)
    final_choice2 = sum(1 for t in balanced_trials if t['choice'] == 2)
    
    print(f"  平衡后的总trials数量: {len(balanced_trials)}")
    print(f"  choice=1: {final_choice1} ({final_choice1/len(balanced_trials)*100:.1f}%)")
    print(f"  choice=2: {final_choice2} ({final_choice2/len(balanced_trials)*100:.1f}%)")
    
    return balanced_trials


def extract_features_for_hanglei(consistent_trials, stage_dataset):
    """
    为hanglei的consistent_trials提取注意力模型所需的所有特征
    由于stage_dataset.pair_names就是consistent_trials中的pair_name，可以直接索引匹配
    """
    print(f"\n=== 提取注意力模型特征 ===")
    
    if not hasattr(stage_dataset, "pair_names") or stage_dataset.pair_names is None:
        raise RuntimeError("StageFeatureDataset 缺少 pair_names 信息，无法对齐行为数据")
    
    # 创建pair_name到索引的映射
    pair_to_idx = {name: idx for idx, name in enumerate(stage_dataset.pair_names)}
    
    X_global_list = []
    X_stage_list = []
    left_globals_list = []
    right_globals_list = []
    left_stage_list = []
    right_stage_list = []
    y_choice_list = []
    missing_pairs = []
    
    for trial in consistent_trials:
        pair_name = trial['pair_name']
        
        # 直接在stage_dataset中查找
        if pair_name in pair_to_idx:
            idx = pair_to_idx[pair_name]
            if idx < len(stage_dataset.global_feats):
                X_global_list.append(stage_dataset.global_feats[idx].copy())
                X_stage_list.append(stage_dataset.stage_feats[idx].copy())
                left_globals_list.append(stage_dataset.left_globals[idx].copy())
                right_globals_list.append(stage_dataset.right_globals[idx].copy())
                left_stage_list.append(stage_dataset.stage_left_feats[idx].copy())
                right_stage_list.append(stage_dataset.stage_right_feats[idx].copy())
                y_choice_list.append(trial['choice'])
            else:
                missing_pairs.append(pair_name)
        else:
            missing_pairs.append(pair_name)
    
    if missing_pairs:
        unique_missing = sorted(set(missing_pairs))
        print(f"  警告: {len(unique_missing)} 个 pair 未在特征集中找到，将被忽略")
        preview = unique_missing[:10]
        for name in preview:
            print(f"    - {name}")
        if len(unique_missing) > len(preview):
            print(f"    ... 其余 {len(unique_missing) - len(preview)} 个省略 ...")
    
    if len(X_global_list) == 0:
        print("  警告: 没有提取到任何有效特征")
        return None
    
    subject_data = {
        "X_global": np.stack(X_global_list, axis=0),
        "X_stage": np.stack(X_stage_list, axis=0),
        "left_global": np.stack(left_globals_list, axis=0),
        "right_global": np.stack(right_globals_list, axis=0),
        "left_stage": np.stack(left_stage_list, axis=0),
        "right_stage": np.stack(right_stage_list, axis=0),
        "y_choice": np.asarray(y_choice_list, dtype=int),
    }
    
    print(f"  提取了 {subject_data['X_global'].shape[0]} 个有效试次的特征")
    print(f"  全局特征形状: {subject_data['X_global'].shape}")
    print(f"  阶段特征形状: {subject_data['X_stage'].shape}")
    print(f"  选左: {np.sum(subject_data['y_choice'] == 1)} ({np.mean(subject_data['y_choice'] == 1)*100:.1f}%)")
    print(f"  选右: {np.sum(subject_data['y_choice'] == 2)} ({np.mean(subject_data['y_choice'] == 2)*100:.1f}%)")
    
    return subject_data


def run_attention_models_for_hanglei(consistent_trials, pair_names, output_dir):
    """针对hanglei运行全部注意力模型并保存结果"""
    print(f"\n{'=' * 60}")
    print(f"开始评估hanglei")
    print(f"{'=' * 60}")
    
    reset_feature_caches()
    all_results = []
    
    # 构建平衡的trials（确保标签均匀分布）
    balanced_trials = build_balanced_trials(consistent_trials, pair_names)
    
    # 构建stage_dataset（现场提取）
    stage_dataset = build_stage_dataset_for_trials(balanced_trials, AUDIO_BASE_DIR)
    
    # 提取特征
    subject_data = extract_features_for_hanglei(balanced_trials, stage_dataset)
    
    if subject_data is None or subject_data["X_global"].shape[0] < 10:
        count = 0 if subject_data is None else subject_data["X_global"].shape[0]
        print(f"  警告: 有效试次太少（{count}），跳过")
        return []
    
    for model_idx, model_config in enumerate(ATTENTION_MODEL_CONFIGS, 1):
        model_name = model_config['model_name']
        print(f"\n{'-' * 60}")
        print(f"[hanglei] 模型 {model_idx}/{len(ATTENTION_MODEL_CONFIGS)}: {model_name}")
        print(f"{'-' * 60}")
        
        try:
            result = fit_attention_choice_model(subject_data, model_name)
            
            if result is not None:
                result['model_name'] = model_name
                result['model_idx'] = model_idx
                result['subject_id'] = 'hanglei'
                result['n_samples'] = len(result.get('y_true', []))
                all_results.append(result)
            else:
                all_results.append(None)
        
        except Exception as e:
            print(f"  错误: 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_attention_model_comparison(all_results, output_dir)
    save_attention_results(all_results, balanced_trials, output_dir, subject_id='hanglei')
    
    valid_results = [r for r in all_results if r is not None]
    
    print(f"\n{'=' * 60}")
    print(f"hanglei 注意力模型对比总结")
    print(f"{'=' * 60}")
    if valid_results:
        # 按 CV 准确率排序
        valid_results.sort(key=lambda x: x.get('cv_accuracy_mean', 0), reverse=True)
        print(f"\n成功拟合 {len(valid_results)} 个模型")
        top_k = min(5, len(valid_results))
        print(f"\n前{top_k}名模型（按CV准确率）:")
        for i, result in enumerate(valid_results[:top_k], 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   准确率: {result['accuracy']:.4f}")
            print(f"   基线准确率: {result['baseline_accuracy']:.4f}")
            print(f"   提升: {(result['accuracy'] - result['baseline_accuracy'])*100:.2f}%")
            print(f"   AUC: {result['auc']:.4f}")
            print(f"   CV准确率: {result.get('cv_accuracy_mean', float('nan')):.4f} ± {result.get('cv_accuracy_std', float('nan')):.4f}")
    else:
        print("没有成功拟合的模型")
    
    print(f"\n结果已保存到: {output_dir}")
    
    return valid_results


def main():
    """主函数"""
    print("=" * 60)
    print("使用注意力模型测试hanglei数据")
    print("=" * 60)
    
    project_root = os.getcwd()
    
    # 1. 加载音频映射文件
    mappings, folder_ranges = load_all_mappings()
    
    # 2. 从映射文件中提取所有音频对名称（480个）
    all_original_names = set()
    for mapping in mappings:
        all_original_names.update(mapping.values())
    pair_names = sorted(list(all_original_names))
    print(f"\n从映射文件提取的音频对总数: {len(pair_names)}")
    
    # 3. 加载hanglei的trial信息（三个session）
    all_trials = load_hanglei_trials(
        benchmark_files=BENCHMARK_FILES,
        mappings=mappings,
        folder_ranges=folder_ranges
    )
    
    if len(all_trials) == 0:
        print("\n错误: 未能加载任何trial信息")
        return
    
    # 4. 筛选一致的trials
    consistent_trials = check_left_right_consistency(all_trials, pair_names)
    
    if len(consistent_trials) == 0:
        print("\n错误: 没有一致的trials")
        return
    
    print(f"\n一致的trials数量: {len(consistent_trials)}")
    
    # 5. 运行所有注意力模型（使用平衡的trials）
    results = run_attention_models_for_hanglei(consistent_trials, pair_names, OUTPUT_DIR)
    
    if results:
        print(f"\n共成功拟合 {len(results)} 个模型")
    else:
        print("\n未生成任何模型结果")


if __name__ == '__main__':
    main()

