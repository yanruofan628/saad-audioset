import os
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment


def extract_first_2s(audio_path, target_duration=2.0):
    """
    从音频文件中提取前2秒

    参数:
    audio_path: 音频文件路径
    target_duration: 目标时长（秒），默认2秒

    返回:
    first_2s: 前2秒音频，如果处理失败返回None
    """
    try:
        # 加载音频文件
        audio = AudioSegment.from_file(audio_path)
        
        # 获取音频时长（毫秒）
        duration_ms = len(audio)
        duration_s = duration_ms / 1000.0
        
        target_duration_ms = int(target_duration * 1000)
        
        # 提取前2秒
        first_2s = audio[:target_duration_ms]
        
        return first_2s
        
    except Exception as e:
        print(f"    ✗ 处理音频失败: {str(e)}")
        return None


def find_audio_files_with_suffix(base_path, suffix="_1"):
    """
    递归查找所有以指定后缀结尾的音频文件

    参数:
    base_path: 基础搜索路径
    suffix: 文件名后缀（默认 "_1"）

    返回:
    audio_files: 包含 (相对路径, 完整路径) 的列表
    """
    audio_files = []
    
    # 支持的音频文件格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    # 遍历基础路径下的所有文件
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            # 检查文件是否以指定后缀结尾
            base_name = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # 检查是否是音频文件且以指定后缀结尾
            if file_ext in audio_extensions and base_name.endswith(suffix):
                # 获取完整路径
                full_path = os.path.join(root, file_name)
                # 获取相对于基础路径的相对路径
                rel_path = os.path.relpath(root, base_path)
                audio_files.append((rel_path, file_name, full_path))
    
    return audio_files


def process_audio_files(input_base_path, output_base_dir):
    """
    处理所有以_1结尾的音频文件，提取前2秒并保持目录结构

    参数:
    input_base_path: 输入文件的基础路径
    output_base_dir: 输出基础目录
    """
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 查找所有以_1结尾的音频文件
    print(f"在 {input_base_path} 中查找以 '_1' 结尾的音频文件...")
    audio_files = find_audio_files_with_suffix(input_base_path, suffix="_1")
    print(f"找到 {len(audio_files)} 个音频文件")
    
    if not audio_files:
        print("未找到符合条件的音频文件")
        return
    
    # 处理每个音频文件
    success_count = 0
    failed_count = 0
    
    for rel_path, file_name, full_path in tqdm(audio_files, desc="处理音频文件"):
        try:
            # 提取前2秒
            first_2s = extract_first_2s(full_path)
            
            if first_2s is None:
                failed_count += 1
                continue
            
            # 构建输出路径，保持目录结构
            if rel_path == '.':
                # 如果文件在根目录
                output_dir = output_base_dir
            else:
                # 保持相对路径结构
                output_dir = os.path.join(output_base_dir, rel_path)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建输出文件路径（保持原文件名）
            output_path = os.path.join(output_dir, file_name)
            
            # 获取文件扩展名（去掉点）
            file_ext = os.path.splitext(file_name)[1]
            format_name = file_ext[1:] if file_ext.startswith('.') else file_ext
            
            # 保存音频文件
            first_2s.export(output_path, format=format_name)
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ 处理文件失败 {file_name}: {str(e)}")
            failed_count += 1
    
    # 输出统计信息
    print(f"\n=== 处理完成统计 ===")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"输出目录: {output_base_dir}")


def main():
    """
    主函数：从clap_select文件夹中提取所有_1结尾音频文件的前2秒
    """
    print("=== 音频2秒切片提取工具 ===")
    
    # 设置路径
    input_base_path = r"D:\D\research\audioset下载\clap_select"  # 输入文件的基础路径
    output_base_dir = r"D:\D\research\audioset下载\clap_select_2s"  # 输出目录
    
    # 检查输入路径是否存在
    if not os.path.exists(input_base_path):
        print(f"错误: 输入路径不存在: {input_base_path}")
        return
    
    # 处理音频文件
    process_audio_files(input_base_path, output_base_dir)
    
    # 统计输出目录中的文件数量
    print(f"\n=== 输出目录统计 ===")
    total_files = 0
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    if os.path.exists(output_base_dir):
        for root, dirs, files in os.walk(output_base_dir):
            files_count = len([f for f in files 
                             if any(f.lower().endswith(ext) for ext in audio_extensions)])
            total_files += files_count
        
        print(f"输出目录中共有: {total_files} 个音频文件")


if __name__ == "__main__":
    main()

