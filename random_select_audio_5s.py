import os
import random
import shutil
from pathlib import Path

def random_select_audio_5s(source_base_folder, output_folder, selected_count=60):
    """
    从音频配对合成_5s文件夹中随机选择音频文件
    
    参数:
    source_base_folder: 源文件夹路径 (D:\D\research\audioset下载\音频配对合成_5s)
    output_folder: 输出文件夹路径
    selected_count: 每个类别选择的文件数量，默认112
    """
    print("=== 从音频配对合成_5s中随机选择音频文件 ===")
    
    # 支持的音频文件格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建子文件夹
    selected_folder = "selected_120_mixed"  # 选中的120条音频（打乱）
    remaining_folder = "remaining_all_mixed"  # 剩余的所有音频（打乱）
    output_selected = os.path.join(output_folder, selected_folder)
    output_remaining = os.path.join(output_folder, remaining_folder)
    os.makedirs(output_selected, exist_ok=True)
    os.makedirs(output_remaining, exist_ok=True)
    
    # 源文件夹路径
    nearest_neighbor_folder = os.path.join(source_base_folder, "nearest_neighbor_pairs")
    random_pairs_folder = os.path.join(source_base_folder, "random_pairs")
    
    print(f"源基础文件夹: {source_base_folder}")
    print(f"最近邻配对文件夹: {nearest_neighbor_folder}")
    print(f"随机配对文件夹: {random_pairs_folder}")
    print(f"选中文件输出: {output_selected}")
    print(f"剩余文件输出: {output_remaining}")
    print(f"每个类别选择: {selected_count} 个文件")
    
    def get_audio_files(folder_path):
        """获取文件夹中的所有音频文件"""
        audio_files = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)
        return audio_files
    
    def copy_files_with_prefix(source_folder, file_list, output_folder, prefix):
        """复制文件列表到输出文件夹，并添加前缀"""
        copied_files = []
        for i, file_name in enumerate(file_list, 1):
            source_path = os.path.join(source_folder, file_name)
            # 添加前缀以区分来源
            new_filename = f"{prefix}_{file_name}"
            output_path = os.path.join(output_folder, new_filename)
            
            try:
                shutil.copy2(source_path, output_path)
                copied_files.append(new_filename)
                print(f"  {i:3d}. ✓ 复制: {file_name} -> {new_filename}")
            except Exception as e:
                print(f"  {i:3d}. ✗ 复制失败: {file_name} - {e}")
        
        return copied_files
    
    # 检查源文件夹是否存在
    if not os.path.exists(nearest_neighbor_folder):
        print(f"错误: 最近邻配对文件夹不存在: {nearest_neighbor_folder}")
        return None
    
    if not os.path.exists(random_pairs_folder):
        print(f"错误: 随机配对文件夹不存在: {random_pairs_folder}")
        return None
    
    # 获取所有音频文件
    print(f"\n{'='*60}")
    print("=== 扫描音频文件 ===")
    
    nearest_files = get_audio_files(nearest_neighbor_folder)
    random_files = get_audio_files(random_pairs_folder)
    
    print(f"最近邻配对文件夹: 找到 {len(nearest_files)} 个音频文件")
    print(f"随机配对文件夹: 找到 {len(random_files)} 个音频文件")
    print(f"总计: {len(nearest_files) + len(random_files)} 个音频文件")
    
    # 随机选择文件
    print(f"\n{'='*60}")
    print("=== 随机选择文件 ===")
    
    # 从最近邻文件夹选择
    if len(nearest_files) < selected_count:
        print(f"警告: 最近邻配对只有 {len(nearest_files)} 个文件，少于请求的 {selected_count} 个")
        selected_nearest = nearest_files
    else:
        selected_nearest = random.sample(nearest_files, selected_count)
    
    # 从随机配对文件夹选择
    if len(random_files) < selected_count:
        print(f"警告: 随机配对只有 {len(random_files)} 个文件，少于请求的 {selected_count} 个")
        selected_random = random_files
    else:
        selected_random = random.sample(random_files, selected_count)
    
    print(f"从最近邻配对选择了: {len(selected_nearest)} 个文件")
    print(f"从随机配对选择了: {len(selected_random)} 个文件")
    print(f"总共选择了: {len(selected_nearest) + len(selected_random)} 个文件")
    
    # 计算剩余文件
    remaining_nearest = [f for f in nearest_files if f not in selected_nearest]
    remaining_random = [f for f in random_files if f not in selected_random]
    
    print(f"最近邻配对剩余: {len(remaining_nearest)} 个文件")
    print(f"随机配对剩余: {len(remaining_random)} 个文件")
    print(f"总共剩余: {len(remaining_nearest) + len(remaining_random)} 个文件")
    
    # 复制选中的文件（打乱顺序）
    print(f"\n{'='*60}")
    print("=== 复制选中的文件（打乱顺序）===")
    
    # 合并选中的文件并打乱
    all_selected = selected_nearest + selected_random
    random.shuffle(all_selected)
    
    print(f"复制 {len(all_selected)} 个选中的文件到 {selected_folder} 文件夹...")
    
    selected_copied = []
    for i, file_name in enumerate(all_selected, 1):
        # 判断文件来源
        if file_name in selected_nearest:
            source_folder = nearest_neighbor_folder
            prefix = "nearest"
        else:
            source_folder = random_pairs_folder
            prefix = "random"
        
        source_path = os.path.join(source_folder, file_name)
        new_filename = f"{prefix}_{file_name}"
        output_path = os.path.join(output_selected, new_filename)
        
        try:
            shutil.copy2(source_path, output_path)
            selected_copied.append(new_filename)
            print(f"  {i:3d}. ✓ 复制: {file_name} -> {new_filename}")
        except Exception as e:
            print(f"  {i:3d}. ✗ 复制失败: {file_name} - {e}")
    
    # 复制剩余的文件（打乱顺序）
    print(f"\n{'='*60}")
    print("=== 复制剩余的文件（打乱顺序）===")
    
    # 合并剩余的文件并打乱
    all_remaining = remaining_nearest + remaining_random
    random.shuffle(all_remaining)
    
    print(f"复制 {len(all_remaining)} 个剩余的文件到 {remaining_folder} 文件夹...")
    
    remaining_copied = []
    for i, file_name in enumerate(all_remaining, 1):
        # 判断文件来源
        if file_name in remaining_nearest:
            source_folder = nearest_neighbor_folder
            prefix = "nearest"
        else:
            source_folder = random_pairs_folder
            prefix = "random"
        
        source_path = os.path.join(source_folder, file_name)
        new_filename = f"{prefix}_{file_name}"
        output_path = os.path.join(output_remaining, new_filename)
        
        try:
            shutil.copy2(source_path, output_path)
            remaining_copied.append(new_filename)
            print(f"  {i:3d}. ✓ 复制: {file_name} -> {new_filename}")
        except Exception as e:
            print(f"  {i:3d}. ✗ 复制失败: {file_name} - {e}")
    
    # 输出总结
    print(f"\n{'='*60}")
    print("=== 选择完成 ===")
    print(f"输出文件夹: {output_folder}")
    print(f"{selected_folder}: 复制了 {len(selected_copied)} 个文件")
    print(f"  - 最近邻配对: {len([f for f in selected_copied if f.startswith('nearest_')])} 个")
    print(f"  - 随机配对: {len([f for f in selected_copied if f.startswith('random_')])} 个")
    print(f"{remaining_folder}: 复制了 {len(remaining_copied)} 个文件")
    print(f"  - 最近邻配对: {len([f for f in remaining_copied if f.startswith('nearest_')])} 个")
    print(f"  - 随机配对: {len([f for f in remaining_copied if f.startswith('random_')])} 个")
    print(f"总计: {len(selected_copied) + len(remaining_copied)} 个文件")
    
    return {
        'output_folder': output_folder,
        'selected_folder': {
            'name': selected_folder,
            'files': selected_copied,
            'count': len(selected_copied),
            'nearest_count': len([f for f in selected_copied if f.startswith('nearest_')]),
            'random_count': len([f for f in selected_copied if f.startswith('random_')])
        },
        'remaining_folder': {
            'name': remaining_folder,
            'files': remaining_copied,
            'count': len(remaining_copied),
            'nearest_count': len([f for f in remaining_copied if f.startswith('nearest_')]),
            'random_count': len([f for f in remaining_copied if f.startswith('random_')])
        }
    }

def main():
    """主函数"""
    # 设置路径
    source_base_folder = r"D:\D\research\audioset下载\音频配对合成_5s"
    output_folder = r"D:\D\research\audioset下载\随机选择配对音频_5s"
    selected_count = 60  # 每个类别选择的文件数量
    
    print("=== 从音频配对合成_5s中随机选择音频文件 ===")
    
    # 执行随机选择
    result = random_select_audio_5s(source_base_folder, output_folder, selected_count)
    
    if result:
        print(f"\n✓ 任务完成！")
        print(f"文件已保存到: {result['output_folder']}")
        print(f"选中文件 ({result['selected_folder']['name']}): {result['selected_folder']['count']} 个")
        print(f"剩余文件 ({result['remaining_folder']['name']}): {result['remaining_folder']['count']} 个")

if __name__ == "__main__":
    main()
