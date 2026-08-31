import os
import subprocess

# 测试FFmpeg命令
VIDEO_PATH = r"D:\D\research\audioset下载\RF.mp4"
AUDIO_DIR = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output"

# 获取第一个音频文件
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
if audio_files:
    audio_path = os.path.join(AUDIO_DIR, audio_files[0])
    print(f"测试音频文件: {audio_path}")
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    temp_video_path = os.path.join(OUTPUT_DIR, "test_output.mp4")
    
    # 使用正确的FFmpeg路径
    ffmpeg_path = r"C:\Users\Lenovo\anaconda3\envs\YOLOv8-Deepsort\Library\bin\ffmpeg.exe"
    
    cmd = [
        ffmpeg_path,
        '-i', VIDEO_PATH,
        '-i', audio_path,
        '-filter_complex',
        '[1:a]adelay=5000|5000[delayed_audio];[delayed_audio]atrim=duration=10[trimmed_audio];[trimmed_audio]apad=pad_dur=2[final_audio]',
        '-map', '0:v',
        '-map', '[final_audio]',
        '-t', '17',
        '-y',
        temp_video_path
    ]
    
    print(f"FFmpeg命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        print("FFmpeg执行成功!")
        print(f"输出文件: {temp_video_path}")
        
        if os.path.exists(temp_video_path):
            file_size = os.path.getsize(temp_video_path)
            print(f"文件大小: {file_size} 字节")
        else:
            print("错误: 输出文件未创建")
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg错误: {e}")
        print(f"stderr: {e.stderr.decode()}")
        print(f"stdout: {e.stdout.decode()}")
    except Exception as e:
        print(f"其他错误: {e}")
else:
    print("没有找到音频文件")
