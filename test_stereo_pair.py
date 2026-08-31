import os
import librosa
import soundfile as sf
import numpy as np

def create_stereo_audio(left_audio_path, right_audio_path, output_path, target_duration=2.0, sr=16000):
    """
    创建左右声道分离的双声道音频
    
    参数:
    left_audio_path: 左声道音频路径
    right_audio_path: 右声道音频路径
    output_path: 输出文件路径
    target_duration: 目标时长（秒），默认2秒
    sr: 采样率，默认16kHz
    """
    try:
        # 加载左声道音频
        left_audio, sr_left = librosa.load(left_audio_path, sr=sr, mono=True)
        print(f"左声道: {os.path.basename(left_audio_path)}")
        print(f"  时长: {len(left_audio)/sr_left:.2f}秒, 采样率: {sr_left}Hz")
        
        # 加载右声道音频
        right_audio, sr_right = librosa.load(right_audio_path, sr=sr, mono=True)
        print(f"右声道: {os.path.basename(right_audio_path)}")
        print(f"  时长: {len(right_audio)/sr_right:.2f}秒, 采样率: {sr_right}Hz")
        
        # 确保是单声道
        if len(left_audio.shape) > 1:
            left_audio = left_audio.flatten()
        if len(right_audio.shape) > 1:
            right_audio = right_audio.flatten()
        
        # 计算目标样本数
        target_samples = int(target_duration * sr)
        
        # 处理左声道：截断或填充到2秒
        left_len = len(left_audio)
        if left_len > target_samples:
            left_audio = left_audio[:target_samples]
            print(f"  左声道截断到 {target_duration} 秒")
        elif left_len < target_samples:
            left_audio = np.pad(left_audio, (0, target_samples - left_len), mode='constant')
            print(f"  左声道填充到 {target_duration} 秒")
        
        # 处理右声道：截断或填充到2秒
        right_len = len(right_audio)
        if right_len > target_samples:
            right_audio = right_audio[:target_samples]
            print(f"  右声道截断到 {target_duration} 秒")
        elif right_len < target_samples:
            right_audio = np.pad(right_audio, (0, target_samples - right_len), mode='constant')
            print(f"  右声道填充到 {target_duration} 秒")
        
        # 创建左右声道分离的立体声数组
        stereo_audio = np.column_stack((left_audio, right_audio))
        
        # 保存为WAV文件
        sf.write(output_path, stereo_audio, sr)
        print(f"\n✓ 成功创建立体声音频: {output_path}")
        print(f"  左声道: {os.path.basename(left_audio_path)}")
        print(f"  右声道: {os.path.basename(right_audio_path)}")
        
        return True
    except Exception as e:
        print(f"✗ 创建立体声音频失败: {e}")
        return False


def main():
    # clap_select_2s 文件夹路径
    base_path = r"D:\D\research\audioset下载\clap_select_2s"
    
    # 查找所有音频文件
    audio_files = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    if len(audio_files) < 2:
        print("错误: 至少需要2个音频文件")
        return
    
    # 选择两个不同类别的音频（如果可能）
    # 优先选择不同子文件夹的音频
    audio1 = audio_files[0]
    audio2 = None
    
    # 尝试找不同类别的音频
    audio1_category = os.path.dirname(audio1)
    for audio in audio_files[1:]:
        if os.path.dirname(audio) != audio1_category:
            audio2 = audio
            break
    
    # 如果没找到不同类别的，就用第二个文件
    if audio2 is None:
        audio2 = audio_files[1]
    
    print("=" * 60)
    print("选择的音频文件:")
    print("=" * 60)
    print(f"音频1 (左声道): {audio1}")
    print(f"  类别: {os.path.basename(os.path.dirname(audio1))}")
    print(f"\n音频2 (右声道): {audio2}")
    print(f"  类别: {os.path.basename(os.path.dirname(audio2))}")
    print("=" * 60)
    
    # 创建输出文件路径
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    audio1_name = os.path.splitext(os.path.basename(audio1))[0]
    audio2_name = os.path.splitext(os.path.basename(audio2))[0]
    output_filename = f"stereo_{audio1_name}+{audio2_name}.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    # 合成立体声音频
    print("\n开始合成立体声音频...")
    success = create_stereo_audio(audio1, audio2, output_path, target_duration=2.0)
    
    if success:
        print(f"\n✓ 完成！音频文件已保存到: {os.path.abspath(output_path)}")
        print(f"\n提示: 使用耳机或立体声音箱播放，可以听到:")
        print(f"  - 左声道: {os.path.basename(audio1)}")
        print(f"  - 右声道: {os.path.basename(audio2)}")


if __name__ == "__main__":
    main()

