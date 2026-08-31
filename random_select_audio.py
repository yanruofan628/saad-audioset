import os
import random
import shutil
from pathlib import Path

def random_select_audio(source_folder1, source_folder2, output_folder, num_files=60):
    """
    从两个源文件夹中各自随机选择指定数量的音频文件，复制到输出文件夹
    
    参数:
    source_folder1: 第一个源文件夹路径
    source_folder2: 第二个源文件夹路径  
    output_folder: 输出文件夹路径
    num_files: 每个文件夹选择的文件数量，默认60
    """
    print("=== 随机选择音频文件 ===")
    
    # 支持的音频文件格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建子文件夹
    folder1_name = "nearest_neighbor_pairs"  # 最近邻配对
    folder2_name = "random_pairs"            # 随机配对
    output_folder1 = os.path.join(output_folder, folder1_name)
    output_folder2 = os.path.join(output_folder, folder2_name)
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    def get_audio_files(folder_path):
        """获取文件夹中的所有音频文件"""
        audio_files = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)
        return audio_files
    
    def copy_random_files(source_folder, output_folder, num_files, folder_name):
        """从源文件夹随机选择文件复制到输出文件夹"""
        print(f"\n处理文件夹: {folder_name}")
        print(f"源路径: {source_folder}")
        print(f"输出路径: {output_folder}")
        
        # 获取所有音频文件
        audio_files = get_audio_files(source_folder)
        print(f"找到 {len(audio_files)} 个音频文件")
        
        if len(audio_files) == 0:
            print(f"警告: {folder_name} 中没有找到音频文件")
            return []
        
        # 随机选择文件
        if len(audio_files) < num_files:
            print(f"警告: {folder_name} 只有 {len(audio_files)} 个文件，少于请求的 {num_files} 个")
            selected_files = audio_files
        else:
            selected_files = random.sample(audio_files, num_files)
        
        print(f"随机选择了 {len(selected_files)} 个文件")
        
        # 复制文件
        copied_files = []
        for i, file_name in enumerate(selected_files, 1):
            source_path = os.path.join(source_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            try:
                shutil.copy2(source_path, output_path)
                copied_files.append(file_name)
                print(f"  {i:2d}. ✓ 复制: {file_name}")
            except Exception as e:
                print(f"  {i:2d}. ✗ 复制失败: {file_name} - {e}")
        
        print(f"成功复制 {len(copied_files)} 个文件到 {output_folder}")
        return copied_files
    
    # 处理第一个文件夹
    print(f"\n{'='*50}")
    copied_files1 = copy_random_files(source_folder1, output_folder1, num_files, folder1_name)
    
    # 处理第二个文件夹
    print(f"\n{'='*50}")
    copied_files2 = copy_random_files(source_folder2, output_folder2, num_files, folder2_name)
    
    # 输出总结
    print(f"\n{'='*50}")
    print("=== 选择完成 ===")
    print(f"输出文件夹: {output_folder}")
    print(f"{folder1_name}: 复制了 {len(copied_files1)} 个文件")
    print(f"{folder2_name}: 复制了 {len(copied_files2)} 个文件")
    print(f"总计: {len(copied_files1) + len(copied_files2)} 个文件")
    
    return {
        'output_folder': output_folder,
        'folder1': {
            'name': folder1_name,
            'files': copied_files1,
            'count': len(copied_files1)
        },
        'folder2': {
            'name': folder2_name,
            'files': copied_files2,
            'count': len(copied_files2)
        }
    }

def main():
    """主函数"""
    # 设置路径 - 从音频配对合成结果中选择
    base_folder = r"D:\D\research\audioset下载\音频配对合成_numpy"
    source_folder1 = os.path.join(base_folder, "nearest_neighbor_pairs")  # 最近邻配对文件夹
    source_folder2 = os.path.join(base_folder, "random_pairs")            # 随机配对文件夹
    output_folder = r"D:\D\research\audioset下载\随机选择配对音频"  # 输出文件夹
    num_files = 60  # 每个文件夹选择的文件数量
    
    print("=== 从音频配对合成结果中随机选择音频文件 ===")
    print(f"基础文件夹: {base_folder}")
    print(f"源文件夹1 (最近邻配对): {source_folder1}")
    print(f"源文件夹2 (随机配对): {source_folder2}")
    print(f"输出文件夹: {output_folder}")
    print(f"每个文件夹选择: {num_files} 个文件")
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder1):
        print(f"错误: 最近邻配对文件夹不存在: {source_folder1}")
        return
    
    if not os.path.exists(source_folder2):
        print(f"错误: 随机配对文件夹不存在: {source_folder2}")
        return
    
    # 执行随机选择
    result = random_select_audio(source_folder1, source_folder2, output_folder, num_files)
    
    if result:
        print(f"\n✓ 任务完成！")
        print(f"文件已保存到: {result['output_folder']}")

if __name__ == "__main__":
    main()
