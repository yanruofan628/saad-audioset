import os
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import mediainfo


def get_category_mapping():
    """获取类别映射关系"""
    category_mapping = {
        '高生态效度': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        '低生态效度': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'music': ['Bass drum', 'Funny music', 'Sad music'],
        '未知声源': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
    }
    return category_mapping


def split_audio_to_5s(audio_path, target_duration=5.0):
    """
    将音频文件分割成5秒的片段

    参数:
    audio_path: 音频文件路径
    target_duration: 目标时长（秒），默认5秒

    返回:
    first_5s: 前5秒音频
    last_5s: 后5秒音频（如果音频长度>=10秒）
    """
    try:
        # 加载音频文件
        audio = AudioSegment.from_file(audio_path)
        
        # 获取音频时长（毫秒）
        duration_ms = len(audio)
        duration_s = duration_ms / 1000.0
        
        target_duration_ms = int(target_duration * 1000)
        
        # 提取前5秒
        first_5s = audio[:target_duration_ms]
        
        # 如果音频长度足够，提取后5秒
        last_5s = None
        if duration_s >= target_duration * 2:
            # 从最后5秒开始提取
            last_5s = audio[-target_duration_ms:]
        elif duration_s > target_duration:
            # 如果音频长度在5-10秒之间，提取剩余部分
            last_5s = audio[target_duration_ms:]
        
        return first_5s, last_5s
        
    except Exception as e:
        print(f"    ✗ 处理音频失败: {str(e)}")
        return None, None


def process_audio_files_in_category(category_dir, output_base_dir, subcategory_name):
    """
    处理单个类别目录中的所有音频文件，分割成5秒片段

    参数:
    category_dir: 类别目录路径
    output_base_dir: 输出基础目录
    subcategory_name: 子类别名称（如 'Bass drum', 'Funny music' 等）
    """
    # 创建输出目录（使用子类别名称）
    output_category_dir = os.path.join(output_base_dir, subcategory_name)
    os.makedirs(output_category_dir, exist_ok=True)

    # 支持的音频文件格式
    file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']

    # 获取所有音频文件
    files = []
    for file_name in os.listdir(category_dir):
        if any(file_name.lower().endswith(ext) for ext in file_extensions):
            files.append(file_name)

    print(f"  找到 {len(files)} 个音频文件")

    if not files:
        print(f"  子类别 {subcategory_name} 中没有找到音频文件")
        return

    # 处理每个音频文件
    success_count = 0
    for file_name in tqdm(files, desc=f"处理 {subcategory_name}"):
        # 获取文件名（不含扩展名）和扩展名
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1]
        
        # 完整的输入文件路径
        input_path = os.path.join(category_dir, file_name)

        # 分割音频为5秒片段
        first_5s, last_5s = split_audio_to_5s(input_path)
        
        if first_5s is None:
            continue

        # 保存前5秒音频
        first_5s_filename = f"{base_name}_1{file_ext}"
        first_5s_path = os.path.join(output_category_dir, first_5s_filename)
        first_5s.export(first_5s_path, format=file_ext[1:])  # 去掉扩展名前的点
        print(f"    ✓ 保存前5秒音频: {first_5s_filename}")

        # 保存后5秒音频（如果存在）
        if last_5s is not None:
            last_5s_filename = f"{base_name}_2{file_ext}"
            last_5s_path = os.path.join(output_category_dir, last_5s_filename)
            last_5s.export(last_5s_path, format=file_ext[1:])
            print(f"    ✓ 保存后5秒音频: {last_5s_filename}")
        
        success_count += 1
    
    print(f"  成功处理 {success_count}/{len(files)} 个文件")


def find_audio_directories(base_path):
    """
    在基础路径中查找包含音频文件的目录

    参数:
    base_path: 基础搜索路径

    返回:
    audio_dirs: 包含音频文件的目录列表
    """
    audio_dirs = []

    # 支持的音频文件格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']

    # 遍历基础路径下的所有目录
    for root, dirs, files in os.walk(base_path):
        # 检查当前目录是否包含音频文件
        has_audio = any(any(file.lower().endswith(ext) for ext in audio_extensions) for file in files)

        if has_audio:
            # 获取相对于基础路径的目录名
            rel_path = os.path.relpath(root, base_path)
            if rel_path != '.':
                audio_dirs.append((rel_path, root))

    return audio_dirs


def main():
    """
    主函数：音频分割工具（将音频文件分割成5秒片段）
    """
    print("=== 音频分割工具（5秒片段）===")

    # 设置路径
    input_base_path = r"D:\D\research\audioset下载\classified_audio"  # 输入文件的基础路径
    output_base_dir = r"D:\D\research\audioset下载\audio_5s_segments"  # 输出目录

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 获取类别映射
    category_mapping = get_category_mapping()
    print(f"类别映射: {category_mapping}")

    # 获取所有子类别
    all_subcategories = []
    for main_cat, sub_cats in category_mapping.items():
        for sub_cat in sub_cats:
            all_subcategories.append((main_cat, sub_cat))

    print(f"总共有 {len(all_subcategories)} 个子类别")

    # 查找所有包含音频文件的目录
    print(f"\n在 {input_base_path} 中查找音频文件...")
    audio_dirs = find_audio_directories(input_base_path)
    print(f"找到 {len(audio_dirs)} 个包含音频文件的目录")

    # 为每个子类别处理音频文件
    processed_categories = set()

    for main_cat, sub_cat in all_subcategories:
        print(f"\n--- 处理子类别: {sub_cat} ---")

        # 查找匹配的目录
        matching_dirs = []
        for dir_name, dir_path in audio_dirs:
            if sub_cat in dir_name or dir_name == sub_cat:
                matching_dirs.append((dir_name, dir_path))

        if not matching_dirs:
            print(f"  未找到 {sub_cat} 的音频文件目录")
            continue

        # 处理找到的目录
        for dir_name, dir_path in matching_dirs:
            print(f"  处理目录: {dir_name}")
            process_audio_files_in_category(dir_path, output_base_dir, sub_cat)
            processed_categories.add(sub_cat)

    # 输出统计信息
    print(f"\n=== 处理完成统计 ===")
    print(f"已处理的子类别: {len(processed_categories)} 个")
    print(f"输出目录: {output_base_dir}")

    # 检查输出目录中的文件
    total_files = 0
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    for subcategory in processed_categories:
        subcategory_dir = os.path.join(output_base_dir, subcategory)
        if os.path.exists(subcategory_dir):
            files_count = len([f for f in os.listdir(subcategory_dir) 
                             if any(f.endswith(ext) for ext in audio_extensions)])
            total_files += files_count
            print(f"  {subcategory}: {files_count} 个音频片段")

    print(f"总共生成: {total_files} 个音频片段")


if __name__ == "__main__":
    main()
