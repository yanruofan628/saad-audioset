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
from flask import Flask, render_template, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import uuid

# 初始化pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置参数
AUDIO_DIR = r"D:\D\research\audioset下载\随机选择配对音频\new_audio"
VIDEO_PATH = r"D:\D\research\audioset下载\RF.mp4"  # 视频文件路径
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "experiment_results2.json")

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 全局变量
experiment_data = {
    "trials": [],
    "start_time": datetime.now().isoformat(),
    "total_duration": 0,
    "current_trial": 0,
    "is_running": False,
    "participants": {}
}

audio_files = []
audio_order = []
original_order = []

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

# 设置随机种子
random_seed = int(time.time() * 1000) % 2**32
random.seed(random_seed)
print(f"使用随机种子: {random_seed}")

# 随机打乱音频顺序
audio_order = list(range(len(audio_files)))
random.shuffle(audio_order)
original_order = audio_files.copy()

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

# Flask路由
@app.route('/')
def index():
    return render_template('experiment.html')

@app.route('/admin')
def admin():
    """管理员控制页面"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>实验控制台</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .btn {
            padding: 15px 30px;
            font-size: 1.2em;
            margin: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
        .btn-reset { background: #ff9800; color: white; }
        .btn:hover { opacity: 0.8; }
        .status {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        h1 { color: #333; text-align: center; }
        .info { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频实验控制台 🎵</h1>
        
        <div class="info">
            <h3>实验信息</h3>
            <p><strong>音频文件数量:</strong> 120个</p>
            <p><strong>视频文件:</strong> RF.mp4</p>
            <p><strong>输出目录:</strong> experiment_output2</p>
        </div>
        
        <div class="status" id="status">
            实验状态: 未开始
        </div>
        
        <div style="text-align: center;">
            <button class="btn btn-start" onclick="startExperiment()">开始实验</button>
            <button class="btn btn-stop" onclick="stopExperiment()">停止实验</button>
            <button class="btn btn-reset" onclick="resetExperiment()">重置实验</button>
        </div>
        
        <div class="info">
            <h3>访问地址</h3>
            <p><strong>被试者访问:</strong> <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>
            <p><strong>局域网访问:</strong> <a href="http://10.11.8.22:5000" target="_blank">http://10.11.8.22:5000</a></p>
        </div>
        
        <div class="info">
            <h3>实验说明</h3>
            <ol>
                <li>点击"开始实验"按钮启动实验</li>
                <li>实验将自动播放120个音频文件</li>
                <li>每个试验17秒（5s静默 + 10s音频 + 2s静默）</li>
                <li>被试者的选择会自动记录</li>
                <li>数据保存在 experiment_output2 目录</li>
            </ol>
        </div>
    </div>
    
    <script>
        function startExperiment() {
            fetch('/start_experiment', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    updateStatus('实验进行中...');
                })
                .catch(error => {
                    alert('启动失败: ' + error);
                });
        }
        
        function stopExperiment() {
            fetch('/stop_experiment', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    updateStatus('实验已停止');
                });
        }
        
        function resetExperiment() {
            if (confirm('确定要重置实验吗？这将清空所有数据！')) {
                fetch('/reset_experiment', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        updateStatus('实验已重置');
                    });
            }
        }
        
        function updateStatus(status) {
            document.getElementById('status').innerHTML = '实验状态: ' + status;
        }
        
        // 定期更新状态
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.is_running) {
                        updateStatus(`实验进行中 - 当前试验: ${data.current_trial} / ${data.total_trials}`);
                    } else {
                        updateStatus('实验未开始');
                    }
                });
        }, 2000);
    </script>
</body>
</html>
    ''')

@app.route('/status')
def get_status():
    return jsonify({
        'is_running': experiment_data['is_running'],
        'current_trial': experiment_data['current_trial'],
        'total_trials': len(audio_order),
        'audio_files': audio_files
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    """提供音频文件访问"""
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        from flask import send_file
        return send_file(audio_path)
    else:
        return "Audio file not found", 404

@app.route('/video')
def serve_video():
    """提供视频文件访问"""
    if os.path.exists(VIDEO_PATH):
        from flask import send_file
        return send_file(VIDEO_PATH)
    else:
        return "Video file not found", 404

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    print(f'用户连接: {request.sid}')
    # 发送当前实验状态
    emit('status_update', {
        'is_running': experiment_data['is_running'],
        'current_trial': experiment_data['current_trial'],
        'total_trials': len(audio_order)
    })

@socketio.on('participant_choice')
def handle_participant_choice(data):
    """处理被试者的选择"""
    print(f'收到选择: {data}')
    
    # 保存选择数据
    choice_data = {
        'participant_id': data.get('participant_id', request.sid),
        'trial_number': data['trial_number'],
        'audio_file': data['audio_file'],
        'choice': data['choice'],
        'reaction_time': data['reaction_time'],
        'timestamp': datetime.now().isoformat()
    }
    
    # 添加到实验结果
    experiment_data['trials'].append(choice_data)
    
    # 保存到文件
    save_results()
    
    # 广播给所有连接的用户
    emit('choice_recorded', choice_data)

@socketio.on('trial_start')
def handle_trial_start(data):
    """广播试验开始"""
    emit('trial_start', data)

@socketio.on('trial_end')
def handle_trial_end():
    """广播试验结束"""
    emit('trial_end')

# 保存结果函数
def save_results():
    """保存实验结果到文件"""
    # 保存JSON格式
    with open(RESULTS_FILE, "w", encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)
    
    # 保存CSV格式
    csv_file = os.path.join(OUTPUT_DIR, "experiment_results.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['试验编号', '音频文件', '选择', '反应时间(ms)', '被试者ID', '时间戳'])
        
        for trial in experiment_data['trials']:
            writer.writerow([
                trial['trial_number'],
                trial['audio_file'],
                trial['choice'],
                f"{trial['reaction_time']:.2f}",
                trial['participant_id'],
                trial['timestamp']
            ])
    
    # 保存TXT格式
    txt_file = os.path.join(OUTPUT_DIR, "experiment_results.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("音频实验结果报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"实验开始时间: {experiment_data['start_time']}\n")
        f.write(f"实验结束时间: {experiment_data.get('end_time', '未完成')}\n")
        f.write(f"总持续时间: {experiment_data.get('total_duration', 0):.2f} 秒\n")
        f.write(f"总试验数: {len(experiment_data['trials'])}\n\n")
        
        f.write("试验详情:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'试验':<6} {'音频文件':<30} {'选择':<8} {'反应时间(ms)':<12} {'被试者ID':<15} {'时间戳'}\n")
        f.write("-" * 80 + "\n")
        
        for trial in experiment_data['trials']:
            f.write(f"{trial['trial_number']:<6} {trial['audio_file']:<30} {trial['choice']:<8} {trial['reaction_time']:<12.2f} {trial['participant_id']:<15} {trial['timestamp']}\n")

# 实验主循环
def run_experiment():
    """运行实验主循环"""
    global experiment_data
    
    experiment_data['is_running'] = True
    experiment_data['current_trial'] = 0
    
    # 广播实验开始
    socketio.emit('experiment_start', {
        'total_trials': len(audio_order),
        'start_time': experiment_data['start_time']
    })
    
    print(f"开始Web实验，共 {len(audio_order)} 个试验")
    
    for i, audio_idx in enumerate(audio_order):
        if not experiment_data['is_running']:
            break
        
        trial_number = i + 1
        experiment_data['current_trial'] = trial_number
        
        # 获取当前音频文件
        current_audio = audio_files[audio_idx]
        
        print(f"\n开始试验 {trial_number}/{len(audio_order)}: {current_audio}")
        
        # 广播试验开始
        trial_data = {
            'trial_number': trial_number,
            'audio_file': current_audio,
            'audio_path': f'/audio/{current_audio}',
            'total_trials': len(audio_order)
        }
        
        socketio.emit('trial_start', trial_data)
        
        # 等待17秒（5s静默 + 10s音频 + 2s静默）
        print("等待17秒...")
        time.sleep(17)
        
        # 广播试验结束
        socketio.emit('trial_end')
        print(f"试验 {trial_number} 结束")
        
        # 短暂暂停
        time.sleep(1)
    
    # 实验结束
    experiment_data['is_running'] = False
    experiment_data['end_time'] = datetime.now().isoformat()
    experiment_data['total_duration'] = time.time() - time.mktime(datetime.fromisoformat(experiment_data['start_time']).timetuple())
    
    # 保存最终结果
    save_results()
    
    # 广播实验结束
    socketio.emit('experiment_end', {
        'end_time': experiment_data['end_time'],
        'total_duration': experiment_data['total_duration'],
        'total_trials': len(experiment_data['trials'])
    })
    
    print("实验完成！")
    print(f"结果保存在: {OUTPUT_DIR}")

# 启动实验的API
@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """启动实验"""
    if experiment_data['is_running']:
        return jsonify({'error': '实验已在运行中'}), 400
    
    # 在新线程中运行实验
    experiment_thread = threading.Thread(target=run_experiment)
    experiment_thread.daemon = True
    experiment_thread.start()
    
    return jsonify({'message': '实验已开始'})

@app.route('/stop_experiment', methods=['POST'])
def stop_experiment():
    """停止实验"""
    experiment_data['is_running'] = False
    return jsonify({'message': '实验已停止'})

@app.route('/reset_experiment', methods=['POST'])
def reset_experiment():
    """重置实验"""
    global experiment_data
    experiment_data = {
        "trials": [],
        "start_time": datetime.now().isoformat(),
        "total_duration": 0,
        "current_trial": 0,
        "is_running": False,
        "participants": {}
    }
    return jsonify({'message': '实验已重置'})

if __name__ == "__main__":
    print("=" * 60)
    print("音频实验 Web 服务器")
    print("=" * 60)
    print(f"音频目录: {AUDIO_DIR}")
    print(f"视频文件: {VIDEO_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"找到 {len(audio_files)} 个音频文件")
    print("=" * 60)
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("管理界面: http://localhost:5000/status")
    print("=" * 60)
    
    try:
        # 启动Flask服务器
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
    finally:
        pygame.quit()
        print("程序结束")
