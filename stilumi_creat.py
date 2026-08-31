import os
import random
import time
import json
import csv
import hashlib
from datetime import datetime
import subprocess
import pygame
import cv2
import numpy as np

# 初始化pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# 配置参数
AUDIO_DIR = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"
VIDEO_PATH = r"D:\D\research\audioset下载\RF.mp4"  # 视频文件路径
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "experiment_results2.json")

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 获取所有音频文件
try:
    if not os.path.exists(AUDIO_DIR):
        print(f"错误: 音频目录不存在: {AUDIO_DIR}")
        exit(1)
    
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
    if len(audio_files) == 0:
        print(f"错误: 在目录 {AUDIO_DIR} 中没有找到音频文件")
        exit(1)
    
    if len(audio_files) != 120:
        print(f"警告: 找到 {len(audio_files)} 个音频文件，但需要120个")
except Exception as e:
    print(f"错误: 无法读取音频目录: {e}")
    exit(1)

# 设置随机种子为当前时间，确保每次运行都有不同的顺序
random_seed = int(time.time() * 1000) % 2**32
random.seed(random_seed)
print(f"使用随机种子: {random_seed}")

# 随机打乱音频顺序但保留原始顺序信息
audio_order = list(range(len(audio_files)))
random.shuffle(audio_order)
original_order = audio_files.copy()  # 保存原始顺序

print(f"音频播放顺序: {audio_order[:10]}...")  # 显示前10个音频的播放顺序

# 保存顺序信息
order_info = {
    "random_seed": random_seed,
    "shuffled_indices": audio_order,
    "original_files": original_order,
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "audio_order.json"), "w") as f:
    json.dump(order_info, f, indent=4)

print("音频顺序已保存到 audio_order.json")

# 设置pygame窗口
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Audio Experiment")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 定义按钮
left_button = pygame.Rect(screen_width // 4 - 100, screen_height // 2 - 50, 200, 100)
right_button = pygame.Rect(3 * screen_width // 4 - 100, screen_height // 2 - 50, 200, 100)

# 实验数据存储
experiment_data = {
    "trials": [],
    "start_time": datetime.now().isoformat(),
    "total_duration": 0
}

# 保存结果到txt文件
def save_results_txt(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("音频实验结果报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"实验开始时间: {data['start_time']}\n")
        f.write(f"实验结束时间: {data.get('end_time', '未完成')}\n")
        f.write(f"总持续时间: {data.get('total_duration', 0):.2f} 秒\n")
        f.write(f"总试验数: {len(data['trials'])}\n\n")
        
        f.write("试验详情:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'试验':<6} {'音频文件':<30} {'选择':<8} {'反应时间(ms)':<12} {'时间戳'}\n")
        f.write("-" * 80 + "\n")
        
        for trial in data['trials']:
            f.write(f"{trial['trial_number']:<6} {trial['audio_file']:<30} {trial['choice']:<8} {trial['reaction_time_ms']:<12.2f} {trial['timestamp']}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("实验完成\n")

# 保存结果到csv文件
def save_results_csv(data, filepath):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['试验编号', '音频文件', '原始索引', '选择', '反应时间(ms)', '时间戳'])
        
        # 写入数据
        for trial in data['trials']:
            writer.writerow([
                trial['trial_number'],
                trial['audio_file'],
                trial['original_index'],
                trial['choice'],
                f"{trial['reaction_time_ms']:.2f}",
                trial['timestamp']
            ])


# 注意：process_trial 函数已删除，现在直接播放原始视频文件


# 显示选项界面并记录反应时间
def show_options(trial_count):
    print(f"开始显示选项界面 - 试验 {trial_count}")
    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 48)

    # 绘制选项文本
    text = font.render("Please choose:", True, BLACK)
    screen.blit(text, (screen_width // 2 - 80, screen_height // 4))

    # 绘制左右按钮
    pygame.draw.rect(screen, BLUE, left_button)
    pygame.draw.rect(screen, RED, right_button)

    left_text = font.render("Left", True, WHITE)
    right_text = font.render("Right", True, WHITE)

    screen.blit(left_text, (left_button.centerx - 20, left_button.centery - 15))
    screen.blit(right_text, (right_button.centerx - 20, right_button.centery - 15))

    pygame.display.flip()
    print("选项界面已显示，等待用户点击...")

    # 等待用户选择
    choice = None
    start_time = time.time() * 1000  # 毫秒
    reaction_time = None

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("用户关闭窗口")
                return None, None
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(f"鼠标点击位置: {event.pos}")
                if left_button.collidepoint(event.pos):
                    choice = "left"
                    reaction_time = time.time() * 1000 - start_time
                    waiting = False
                    print("用户选择了左")
                elif right_button.collidepoint(event.pos):
                    choice = "right"
                    reaction_time = time.time() * 1000 - start_time
                    waiting = False
                    print("用户选择了右")

    return choice, reaction_time


# 主实验循环 - 连续播放版本
def run_experiment():
    trial_count = 0
    running = True
    
    # 打开主视频文件，连续播放
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    frame_interval = 1.0 / fps
    print(f"视频FPS: {fps}, 帧间隔: {frame_interval:.4f}秒")
    
    # 实验开始时间
    experiment_start_time = time.perf_counter()
    current_trial_start_time = experiment_start_time
    
    print(f"开始连续播放实验，共 {len(audio_order)} 个试验")
    print("视频将连续播放，每隔17秒插入一个音频试验")

    for audio_idx in audio_order:
        if not running:
            break

        trial_count += 1
        print(f"\n开始试验 {trial_count}/{len(audio_order)}")
        print(f"当前视频时间: {(current_trial_start_time - experiment_start_time):.1f}秒")

        # 加载当前试验的音频
        try:
            original_audio_path = os.path.join(AUDIO_DIR, audio_files[audio_idx])
            print(f"加载音频文件: {original_audio_path}")
            pygame.mixer.music.load(original_audio_path)
            print(f"音频文件加载成功")
        except Exception as e:
            print(f"加载音频文件时出错: {e}")
            continue
        
        # 播放当前试验段（17秒：5秒静默+10秒音频+2秒静默）
        trial_start_time = time.perf_counter()
        audio_played = False
        audio_stop_scheduled = False
        
        print(f"开始播放试验 {trial_count}: {audio_files[audio_idx]}")
        print("时间安排: 0-5s静默, 5-15s音频, 15-17s静默")

        # 播放17秒的试验段
        while (time.perf_counter() - trial_start_time) < 17.0:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束，实验终止")
                running = False
                break

            # 显示视频帧
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()
            except Exception as e:
                print(f"显示视频帧时出错: {e}")
                break

            # 音频控制：5s 开始，15s 停止
            current_time = time.perf_counter() - trial_start_time
            if not audio_played and current_time >= 5.0:
                try:
                    pygame.mixer.music.play()
                    print(f"开始播放音频 (t={current_time:.2f}s)")
                    audio_played = True
                except Exception as e:
                    print(f"播放音频时出错: {e}")

            if audio_played and not audio_stop_scheduled and current_time >= 15.0:
                pygame.mixer.music.stop()
                audio_stop_scheduled = True
                print(f"音频停止 (t={current_time:.2f}s)")

            # 每2秒显示一次状态
            if int(current_time) % 2 == 0 and current_time > 0:
                print(f"试验播放中... {current_time:.1f}s, 音频: {'播放中' if pygame.mixer.music.get_busy() else '未播放'}")

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            # 控制播放速度
            time.sleep(frame_interval)

        # 停止音频播放
        pygame.mixer.music.stop()
        print("试验段播放完成")
        
        if not running:
            break

        # 显示选项界面
        print(f"显示选项界面 - 试验 {trial_count}")
        choice, reaction_time = show_options(trial_count)
        print(f"用户选择: {choice}, 反应时间: {reaction_time}")

        if choice is None:
            break

        # 记录试验数据
        trial_data = {
            "trial_number": trial_count,
            "audio_file": audio_files[audio_idx],
            "original_index": original_order.index(audio_files[audio_idx]),
            "choice": choice,
            "reaction_time_ms": reaction_time,
            "video_time": current_trial_start_time - experiment_start_time,
            "timestamp": datetime.now().isoformat()
        }

        experiment_data["trials"].append(trial_data)
        print(f"试验 {trial_count} 完成: 选择 {choice}, 反应时间 {reaction_time:.2f}ms")

        # 保存中间结果
        with open(RESULTS_FILE, "w") as f:
            json.dump(experiment_data, f, indent=4)
        
        txt_file = os.path.join(OUTPUT_DIR, "experiment_results.txt")
        save_results_txt(experiment_data, txt_file)
        
        csv_file = os.path.join(OUTPUT_DIR, "experiment_results.csv")
        save_results_csv(experiment_data, csv_file)
        
        # 更新下一个试验的开始时间
        current_trial_start_time = time.perf_counter()
    
    # 释放视频资源
    cap.release()

    # 计算总持续时间
    experiment_data["end_time"] = datetime.now().isoformat()
    experiment_data["total_duration"] = time.time() - start_time

    # 保存最终结果（JSON格式）
    with open(RESULTS_FILE, "w") as f:
        json.dump(experiment_data, f, indent=4)
    
    # 保存最终结果（TXT格式）
    txt_file = os.path.join(OUTPUT_DIR, "experiment_results.txt")
    save_results_txt(experiment_data, txt_file)
    
    # 保存最终结果（CSV格式）
    csv_file = os.path.join(OUTPUT_DIR, "experiment_results.csv")
    save_results_csv(experiment_data, csv_file)

    print(f"实验完成! 结果保存在:")
    print(f"  JSON格式: {RESULTS_FILE}")
    print(f"  TXT格式: {txt_file}")
    print(f"  CSV格式: {csv_file}")


if __name__ == "__main__":
    # 添加测试模式
    import sys
    test_mode = "--test" in sys.argv
    audio_only_mode = "--audio-only" in sys.argv
    
    if test_mode:
        print("运行在测试模式下，只处理前3个音频文件")
        audio_order = audio_order[:3]
    
    if audio_only_mode:
        print("运行在纯音频模式下，只播放音频不显示视频")
        # 简化实验流程，只播放音频
        for i, audio_idx in enumerate(audio_order[:3]):  # 只测试前3个
            audio_path = os.path.join(AUDIO_DIR, audio_files[audio_idx])
            print(f"播放音频 {i+1}: {audio_files[audio_idx]}")
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                print("音频开始播放...")
                pygame.time.wait(10000)  # 播放10秒
                pygame.mixer.music.stop()
                print("音频播放结束")
            except Exception as e:
                print(f"播放音频失败: {e}")
        print("音频测试完成")
        pygame.quit()
        exit(0)
    
    print(f"准备开始实验，共 {len(audio_order)} 个试验")
    print(f"音频目录: {AUDIO_DIR}")
    print(f"视频文件: {VIDEO_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 检查FFmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        print("FFmpeg 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: FFmpeg 不可用，请确保已安装并添加到PATH")
        exit(1)
    
    # 检查pygame音频系统
    print(f"pygame mixer 初始化状态: {pygame.mixer.get_init()}")
    print(f"pygame mixer 是否已初始化: {pygame.mixer.get_init() is not None}")
    
    # 检查音频文件
    if len(audio_files) > 0:
        test_audio_path = os.path.join(AUDIO_DIR, audio_files[0])
        print(f"检查音频文件: {test_audio_path}")
        print(f"文件是否存在: {os.path.exists(test_audio_path)}")
        if os.path.exists(test_audio_path):
            try:
                pygame.mixer.music.load(test_audio_path)
                print("音频文件加载测试成功")
                pygame.mixer.music.stop()  # 立即停止，不播放
            except Exception as e:
                print(f"音频文件加载测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    start_time = time.time()
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("程序结束")