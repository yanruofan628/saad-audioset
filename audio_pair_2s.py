import pandas as pd
import os
import numpy as np
import librosa
import random
from pathlib import Path
import json
from datetime import datetime
import soundfile as sf


def get_category_mapping():
    """获取类别映射关系（4个大类，每个大类2个子类）"""
    category_mapping = {
        '高生态效度': ['Telephone bell ringing', 'Baby cry, infant cry'],
        '低生态效度': ['Computer keyboard', 'Helicopter'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking'],
        'music': ['Bass drum', 'Sad music'],
    }
    return category_mapping


def load_audio_files(clap_audio_path):
    """从各个子类别目录加载音频文件"""
    audio_data = {'categories': {}}
    
    category_mapping = get_category_mapping()
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((main_cat, sub_cat))
    
    print(f"开始从子类别目录加载音频文件...")
    print(f"音频路径: {clap_audio_path}")
    
    for main_cat, sub_cat in all_subcategories:
        # 构建子类别目录路径
        sub_cat_dir = os.path.join(clap_audio_path, sub_cat)
        print(f"\n检查子类别目录: {sub_cat_dir}")
        
        if os.path.exists(sub_cat_dir):
            # 扫描目录中的所有音频文件
            audio_files = []
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
            
            for file_name in os.listdir(sub_cat_dir):
                if any(file_name.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(sub_cat_dir, file_name)
                    audio_files.append({
                        'file_name': file_name,
                        'file_path': file_path,
                        'main_category': main_cat,
                        'sub_category': sub_cat
                    })
            
            if sub_cat not in audio_data['categories']:
                audio_data['categories'][sub_cat] = []
            audio_data['categories'][sub_cat] = audio_files
            
            print(f"  成功加载 {len(audio_files)} 个音频文件")
        else:
            print(f"  警告: 目录不存在 {sub_cat_dir}")
            audio_data['categories'][sub_cat] = []
    
    # 打印统计信息
    print(f"\n=== 音频文件加载统计 ===")
    for main_cat, sub_cats in category_mapping.items():
        print(f"{main_cat}:")
        for sub_cat in sub_cats:
            count = len(audio_data['categories'].get(sub_cat, []))
            print(f"  {sub_cat}: {count} 个文件")
    
    return audio_data


def load_audio_file(audio_path, target_sr=16000):
    """加载音频文件，重采样到16kHz，确保为单声道"""
    try:
        # 使用librosa加载，重采样到16kHz
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 检查是否为单声道
        if len(y.shape) > 1:
            print(f"警告: {os.path.basename(audio_path)} 是多声道，已转换为单声道")
            y = y[:, 0] if len(y.shape) > 1 else y
        
        return y, sr
    except Exception as e:
        print(f"加载音频文件失败 {audio_path}: {e}")
        return None, None


def create_stereo_audio(left_audio, right_audio, output_path, sr=16000, target_duration=2.0):
    """
    创建左右声道分离的双声道音频，确保两个声道都是2秒
    
    参数:
    left_audio: 左声道音频数据（第一个音频，单声道）
    right_audio: 右声道音频数据（第二个音频，单声道）
    output_path: 输出文件路径
    sr: 采样率
    target_duration: 目标时长（秒），默认2秒
    """
    try:
        # 确保输入是单声道
        if len(left_audio.shape) > 1:
            left_audio = left_audio.flatten()
        if len(right_audio.shape) > 1:
            right_audio = right_audio.flatten()
        
        # 计算目标样本数（2秒）
        target_samples = int(target_duration * sr)
        
        # 处理左声道
        left_len = len(left_audio)
        if left_len > target_samples:
            left_audio = left_audio[:target_samples]
        elif left_len < target_samples:
            left_audio = np.pad(left_audio, (0, target_samples - left_len), mode='constant')
        
        # 处理右声道
        right_len = len(right_audio)
        if right_len > target_samples:
            right_audio = right_audio[:target_samples]
        elif right_len < target_samples:
            right_audio = np.pad(right_audio, (0, target_samples - right_len), mode='constant')
        
        # 创建左右声道分离的立体声数组
        stereo_audio = np.column_stack((left_audio, right_audio))
        
        # 保存为WAV文件
        sf.write(output_path, stereo_audio, sr)
        
        return True
    except Exception as e:
        print(f"创建立体声音频失败: {e}")
        return False


def generate_subcategory_pairs(category_mapping):
    """
    生成所有跨大类的子类配对组合
    排除同一大类内的配对和自身配对
    
    返回:
    pairs: 配对列表，每个元素是 (sub_cat1, sub_cat2, main_cat1, main_cat2)
    """
    pairs = []
    all_subcategories = []
    
    # 收集所有子类及其所属大类
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((sub_cat, main_cat))
    
    # 生成所有配对
    for i, (sub_cat1, main_cat1) in enumerate(all_subcategories):
        for j, (sub_cat2, main_cat2) in enumerate(all_subcategories):
            # 排除自身配对
            if sub_cat1 == sub_cat2:
                continue
            # 排除同一大类内的配对
            if main_cat1 == main_cat2:
                continue
            # 避免重复配对（A-B和B-A只保留一个，后面会做左右平衡）
            if i < j:
                pairs.append((sub_cat1, sub_cat2, main_cat1, main_cat2))
    
    return pairs


def random_subcategory_matching(audio_data, output_dir, pairs_per_combination=15):
    """
    随机子类别匹配：生成24种跨大类子类配对组合
    
    参数:
    audio_data: 音频数据
    output_dir: 输出目录
    pairs_per_combination: 每种配对组合生成的配对数，默认15个
    """
    print("\n=== 开始随机子类别匹配阶段（2秒音频版本）===")
    
    category_mapping = get_category_mapping()
    
    # 生成所有跨大类的子类配对组合
    pairs = generate_subcategory_pairs(category_mapping)
    print(f"生成的配对组合数: {len(pairs)}")
    print(f"配对组合列表:")
    for idx, (sub1, sub2, main1, main2) in enumerate(pairs, 1):
        print(f"  {idx}. {sub1} ({main1}) × {sub2} ({main2})")
    
    pairs_created = []
    total_audios = 0
    
    # 为每种配对组合创建配对
    for pair_idx, (sub_cat1, sub_cat2, main_cat1, main_cat2) in enumerate(pairs, 1):
        print(f"\n--- 配对组合 {pair_idx}/{len(pairs)}: {sub_cat1} × {sub_cat2} ---")
        
        # 获取两个子类的音频文件
        audios_1 = audio_data['categories'].get(sub_cat1, [])
        audios_2 = audio_data['categories'].get(sub_cat2, [])
        
        if len(audios_1) == 0:
            print(f"  警告: {sub_cat1} 没有可用音频，跳过")
            continue
        if len(audios_2) == 0:
            print(f"  警告: {sub_cat2} 没有可用音频，跳过")
            continue
        
        print(f"  {sub_cat1}: {len(audios_1)} 个音频")
        print(f"  {sub_cat2}: {len(audios_2)} 个音频")
        
        # 确定需要生成的配对数（考虑左右平衡，所以实际需要 pairs_per_combination 对）
        needed_pairs = pairs_per_combination
        
        # 如果音频不足，允许重复使用
        if len(audios_1) < needed_pairs:
            print(f"  警告: {sub_cat1} 音频不足 {needed_pairs} 个，将重复使用")
            audios_1_sampled = random.choices(audios_1, k=needed_pairs)
        else:
            audios_1_sampled = random.sample(audios_1, needed_pairs)
        
        if len(audios_2) < needed_pairs:
            print(f"  警告: {sub_cat2} 音频不足 {needed_pairs} 个，将重复使用")
            audios_2_sampled = random.choices(audios_2, k=needed_pairs)
        else:
            audios_2_sampled = random.sample(audios_2, needed_pairs)
        
        # 创建配对
        for i in range(needed_pairs):
            audio_A = audios_1_sampled[i]
            audio_B = audios_2_sampled[i]
            
            # 加载音频
            audio_A_data, sr = load_audio_file(audio_A['file_path'])
            audio_B_data, sr = load_audio_file(audio_B['file_path'])
            
            if audio_A_data is None or audio_B_data is None:
                print(f"  配对 {i+1}: 音频加载失败")
                continue
            
            # 生成文件名ID（去掉扩展名）
            audio_A_id = os.path.splitext(audio_A['file_name'])[0]
            audio_B_id = os.path.splitext(audio_B['file_name'])[0]
            
            # 配对1: A左B右
            output_filename_1 = f"sub_{sub_cat1}_{audio_A_id}+{sub_cat2}_{audio_B_id}.wav"
            output_path_1 = os.path.join(output_dir, "subcategory_pairs", output_filename_1)
            os.makedirs(os.path.dirname(output_path_1), exist_ok=True)
            
            success_1 = create_stereo_audio(audio_A_data, audio_B_data, output_path_1, sr, target_duration=2.0)
            if success_1:
                pairs_created.append({
                    'type': 'subcategory_random',
                    'left_main_category': main_cat1,
                    'left_sub_category': sub_cat1,
                    'left_file': audio_A['file_name'],
                    'right_main_category': main_cat2,
                    'right_sub_category': sub_cat2,
                    'right_file': audio_B['file_name'],
                    'output_file': output_filename_1,
                    'combination': f"{sub_cat1} × {sub_cat2}"
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-1: {sub_cat1}(左) + {sub_cat2}(右)")
            
            # 配对2: B左A右（左右平衡）
            output_filename_2 = f"sub_{sub_cat2}_{audio_B_id}+{sub_cat1}_{audio_A_id}.wav"
            output_path_2 = os.path.join(output_dir, "subcategory_pairs", output_filename_2)
            
            success_2 = create_stereo_audio(audio_B_data, audio_A_data, output_path_2, sr, target_duration=2.0)
            if success_2:
                pairs_created.append({
                    'type': 'subcategory_random',
                    'left_main_category': main_cat2,
                    'left_sub_category': sub_cat2,
                    'left_file': audio_B['file_name'],
                    'right_main_category': main_cat1,
                    'right_sub_category': sub_cat1,
                    'right_file': audio_A['file_name'],
                    'output_file': output_filename_2,
                    'combination': f"{sub_cat2} × {sub_cat1}"
                })
                total_audios += 1
                print(f"  ✓ 配对 {i+1}-2: {sub_cat2}(左) + {sub_cat1}(右)")
    
    print(f"\n随机子类别匹配完成: 创建了 {total_audios} 条音频")
    return pairs_created


def save_pairing_results(pairs_created, output_dir):
    """保存配对结果"""
    # 保存为JSON文件
    json_path = os.path.join(output_dir, "pairing_results_2s.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pairs_created, f, ensure_ascii=False, indent=2)
    
    # 保存为CSV文件
    df = pd.DataFrame(pairs_created)
    csv_path = os.path.join(output_dir, "pairing_results_2s.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n配对结果已保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def main(clap_audio_path, output_dir, pairs_per_combination=15):
    """
    主函数：2秒音频配对合成
    
    参数:
    clap_audio_path: 音频文件路径（包含各子类别目录）
    output_dir: 输出目录
    pairs_per_combination: 每种配对组合生成的配对数，默认15个
    """
    print("=== 2秒音频配对合成工具 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频文件
    audio_data = load_audio_files(clap_audio_path)
    
    # 生成配对
    pairs = random_subcategory_matching(audio_data, output_dir, pairs_per_combination)
    
    # 保存结果
    save_pairing_results(pairs, output_dir)
    
    # 输出统计信息
    print(f"\n=== 配对完成统计 ===")
    print(f"总配对: {len(pairs)} 对")
    print(f"输出目录: {output_dir}")
    
    # 统计每种组合的配对数
    category_mapping = get_category_mapping()
    pairs_list = generate_subcategory_pairs(category_mapping)
    print(f"\n=== 配对组合统计 ===")
    for sub1, sub2, main1, main2 in pairs_list:
        count = len([p for p in pairs if p['combination'] == f"{sub1} × {sub2}" or p['combination'] == f"{sub2} × {sub1}"])
        print(f"{sub1} × {sub2}: {count} 个配对")


if __name__ == "__main__":
    # 设置路径
    clap_audio_path = r"D:\D\research\audioset下载\clap_select_2s"  # 2秒音频文件路径
    output_dir = r"D:\D\research\audioset下载\audio_pairs_2s"  # 输出目录
    pairs_per_combination = 10  # 每种配对组合生成的配对数（左右平衡后会翻倍，24种组合×10配对×2方向=480个）
    
    # 运行主函数
    main(clap_audio_path, output_dir, pairs_per_combination)

