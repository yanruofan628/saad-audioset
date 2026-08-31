import os
import random
import time
import json
import csv
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
import threading

# Flask应用
app = Flask(__name__)

# 配置参数（沿用现有目录/视频/输出设置）
AUDIO_DIR = r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1"
VIDEO_PATH = r"D:\BaiduNetdiskDownload\29.第二十九回 吕圣人智斗姬无命 佟掌柜火拼展红绫.mp4"
OUTPUT_DIR = r"D:\D\research\audioset下载\experiment_output2"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 全局音频列表（所有用户共享资源池）
if not os.path.exists(AUDIO_DIR):
    raise RuntimeError(f"音频目录不存在: {AUDIO_DIR}")

audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3', '.ogg'))]
if len(audio_files) == 0:
    raise RuntimeError(f"在目录 {AUDIO_DIR} 中没有找到音频文件")

print(f"[MULTIUSER] 发现音频文件 {len(audio_files)} 个")

# 每位用户的独立状态：username -> {trials, current_trial, is_running, trial_completed, start_time, end_time, audio_order, random_seed}
user_data = {}
user_threads = {}
user_thread_locks = threading.Lock()


def init_user_if_needed(username: str):
    username = (username or '').strip()
    if not username:
        return None
    if username not in user_data:
        # 为该用户创建独立的音频顺序与状态（支持断点续做）
        seed = int(time.time() * 1000) % 2**32
        rng = random.Random(seed)

        # 读取该用户既有CSV，恢复已完成记录
        completed_files = set()
        restored_trials = []
        csv_path = os.path.join(OUTPUT_DIR, f"{safe_filename(username)}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header_skipped = False
                    for row in reader:
                        if not header_skipped:
                            header_skipped = True
                            continue
                        if not row:
                            continue
                        # 期望列：['试验编号','音频文件','选择','反应时间(ms)','被试者ID','时间戳']
                        try:
                            trial_number = int(row[0]) if row[0] else 0
                            audio_file = row[1]
                            choice = row[2]
                            reaction_ms = float(row[3]) if row[3] else 0.0
                            participant_id = row[4]
                            timestamp = row[5]
                            restored_trials.append({
                                'trial_number': trial_number,
                                'audio_file': audio_file,
                                'choice': choice,
                                'reaction_time': reaction_ms,
                                'participant_id': participant_id,
                                'timestamp': timestamp
                            })
                            completed_files.add(audio_file)
                        except Exception:
                            continue
                print(f"[MULTIUSER] 恢复 {username} 已有记录 {len(restored_trials)} 条")
            except Exception as e:
                print(f"[MULTIUSER] 恢复 {username} CSV 失败: {e}")

        # 构建剩余待做的音频顺序
        all_indices = list(range(len(audio_files)))
        remaining_indices = [i for i in all_indices if audio_files[i] not in completed_files]
        rng.shuffle(remaining_indices)

        user_data[username] = {
            'trials': restored_trials,  # 启动时就带上历史记录，保存将覆盖写出全部
            'current_trial': 0,
            'is_running': False,
            'trial_completed': {t['trial_number']: True for t in restored_trials},
            'start_time': None,
            'end_time': None,
            'audio_order': remaining_indices,
            'random_seed': seed,
            'trial_offset': len(restored_trials)  # 新试验编号从已有数量之后继续
        }
        print(f"[MULTIUSER] 初始化用户 {username}, 已完成={len(restored_trials)}, 待做={len(remaining_indices)}, seed={seed}")
    return user_data[username]


def safe_filename(name: str) -> str:
    name = (name or 'unknown').strip()
    cleaned = ''.join(c for c in name if c not in '\\/:*?"<>|')
    return cleaned or 'unknown'


def save_results(username: str):
    # 仅保存CSV，按用户名命名
    state = user_data.get(username)
    if not state:
        return
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_filename(username)}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['试验编号', '音频文件', '选择', '反应时间(ms)', '被试者ID', '时间戳'])
        for trial in state['trials']:
            writer.writerow([
                trial['trial_number'],
                trial['audio_file'],
                trial['choice'],
                f"{trial['reaction_time']:.2f}",
                trial['participant_id'],
                trial['timestamp']
            ])


def run_experiment(username: str):
    state = init_user_if_needed(username)
    if not state:
        return

    state['is_running'] = True
    state['current_trial'] = 0
    state['start_time'] = datetime.now().isoformat()
    state['trial_completed'] = {}

    print(f"[MULTIUSER] 开始 {username} 的实验，共 {len(state['audio_order'])} 个试验")

    for i, audio_idx in enumerate(state['audio_order']):
        if not state['is_running']:
            break
        # 试验编号继续累加
        trial_number = state.get('trial_offset', 0) + i + 1
        state['current_trial'] = trial_number
        state['trial_completed'][trial_number] = False
        current_audio = audio_files[audio_idx]
        print(f"[MULTIUSER] {username} 试验 {trial_number}/{len(state['audio_order'])}: {current_audio}")

        # 等待该用户在前端完成选择
        while not state['trial_completed'].get(trial_number, False) and state['is_running']:
            time.sleep(0.1)

        print(f"[MULTIUSER] {username} 试验 {trial_number} 完成")

    state['is_running'] = False
    state['end_time'] = datetime.now().isoformat()
    save_results(username)
    print(f"[MULTIUSER] {username} 实验完成，结果已写入 {OUTPUT_DIR}")


@app.route('/')
def index():
    # 前端：要求用户输入用户名；所有请求都会带上 username
    return render_template_string('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频实验（多用户版）</title>
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .container { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); text-align: center; max-width: 800px; width: 90%; }
        .choice-buttons { margin: 30px 0; display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .choice-btn { padding: 20px 40px; font-size: 1.5em; font-weight: bold; border: none; border-radius: 15px; cursor: pointer; transition: all 0.3s ease; min-width: 150px; letter-spacing: 2px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .left-btn { background: linear-gradient(135deg, #FF6B6B, #EE5A24); color: white; }
        .right-btn { background: linear-gradient(135deg, #4ECDC4, #44A08D); color: white; }
        .video { width: 100%; max-width: 640px; margin: 20px auto; display: block; }
        .audio-info { margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 10px; font-size: 1.1em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频实验（多用户） 🎵</h1>

        <div id="usernameBlock" style="margin: 10px 0; padding: 15px; background: rgba(255,255,255,0.15); border-radius: 10px;">
            <label for="usernameInput">请输入您的用户名：</label>
            <input id="usernameInput" type="text" placeholder="用户名" style="padding: 8px 12px; border-radius: 6px; border: none; margin-left: 8px;">
            <button id="usernameConfirmBtn" onclick="confirmUsername()" class="choice-btn right-btn" style="padding: 8px 16px; font-size: 1em; margin-left: 10px;">确认</button>
            <div style="margin-top: 8px; font-size: 0.9em; opacity: 0.9;">确认后用于独立时间线与CSV命名</div>
        </div>

        <!-- 用户交互提示（启用音频） -->
        <div id="userInteractionPrompt" style="display: block; background: #ff6b6b; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
            <h3>⚠️ 请先点击页面任意位置来启用音频播放</h3>
            <p>现代浏览器要求用户交互后才能播放音频</p>
            <button onclick="enableAudio()" style="background: white; color: #ff6b6b; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer;">点击启用音频</button>
        </div>

        <div id="videoContainer" style="display:none;">
            <video id="videoPlayer" class="video" muted loop>您的浏览器不支持视频播放</video>
        </div>

        <div class="audio-info" id="audioInfo" style="display: none;">
            <div>当前音频: <span id="currentAudio">-</span></div>
            <audio id="audioPlayer" style="width: 100%; margin-top: 10px; display: none;">您的浏览器不支持音频播放</audio>
        </div>

        <div class="choice-buttons" id="choiceButtons" style="display: none;">
            <button class="choice-btn left-btn" id="leftBtn" onclick="makeChoice('left')">← 左</button>
            <button class="choice-btn right-btn" id="rightBtn" onclick="makeChoice('right')">右 →</button>
        </div>

        <div class="timer" id="timer" style="display: none; font-size:1.2em; font-weight:bold; color:#FFD700; margin-top:10px;"></div>
    </div>

    <script>
        let username = '';
        let audioEnabled = false;
        let videoStarted = false;
        let videoPaused = false;
        let lastTrialNumber = 0;
        let audioPlayTimeout = null;
        let audioStopTimeout = null;
        let videoStartOffsetSeconds = 180;
        let videoStartSeeked = false;
        let trialStartTime = 0;
        let isWaitingForChoice = false;
        let currentTrial = 0;

        function confirmUsername() {
            const input = document.getElementById('usernameInput');
            const val = (input.value || '').trim();
            if (!val) { alert('请先输入用户名'); input.focus(); return; }
            username = val;
            document.getElementById('usernameBlock').style.display = 'none';
        }

        function enableAudio() {
            audioEnabled = true;
            document.getElementById('userInteractionPrompt').style.display = 'none';
            const audio = document.getElementById('audioPlayer');
            if (audio) {
                audio.muted = true;
                audio.play().then(() => { audio.pause(); audio.muted = false; }).catch(() => { audio.muted = false; });
            }
        }

        setInterval(checkExperimentStatus, 1000);
        function checkExperimentStatus() {
            if (!username) return; // 需先确认用户名
            fetch('/status?username=' + encodeURIComponent(username))
                .then(r => r.json())
                .then(updateStatus)
                .catch(() => {});
        }

        function startContinuousVideo() {
            let video = document.getElementById('videoPlayer');
            let videoContainer = document.getElementById('videoContainer');
            if (video && videoContainer) {
                videoContainer.style.display = 'block';
                video.muted = true; video.loop = true;
                if (!video.src) { video.src = '/video'; video.load(); }
                if (!videoStartSeeked) {
                    const seekToStart = () => {
                        try {
                            if (video.duration && video.duration > videoStartOffsetSeconds) {
                                video.currentTime = videoStartOffsetSeconds; videoStartSeeked = true;
                            } else { setTimeout(seekToStart, 200); }
                        } catch (e) { setTimeout(seekToStart, 200); }
                    };
                    if (video.readyState >= 1) { seekToStart(); } else { video.addEventListener('loadedmetadata', seekToStart, { once: true }); }
                }
                // 结束时从设定起点重新播放
                const onVideoEnded = () => {
                    try { video.currentTime = Math.min(videoStartOffsetSeconds, Math.max(0, (video.duration || 0) - 0.1)); } catch (e) {}
                    const p = video.play(); if (p) p.catch(()=>{});
                };
                video.removeEventListener('ended', onVideoEnded);
                video.addEventListener('ended', onVideoEnded);

                if (videoPaused || video.paused) { const p = video.play(); if (p) p.catch(()=>{}); videoPaused = false; }
            }
        }

        function updateStatus(data) {
            if (data.is_running) {
                if (!videoStarted) { startContinuousVideo(); videoStarted = true; }
                if (data.is_running && data.current_trial > 0 && data.current_trial <= data.total_trials && data.current_audio) {
                    if (data.current_trial !== lastTrialNumber) { lastTrialNumber = data.current_trial; startNewTrial(data.current_audio); }
                }
            } else {
                // 停止视频但保持进度
                const video = document.getElementById('videoPlayer');
                if (video && !video.paused) { video.pause(); videoPaused = true; }
                document.getElementById('audioInfo').style.display = 'none';
                document.getElementById('choiceButtons').style.display = 'none';
                document.getElementById('timer').style.display = 'none';
            }
        }

        function startNewTrial(audioFile) {
            const audio = document.getElementById('audioPlayer');
            if (audioPlayTimeout) clearTimeout(audioPlayTimeout);
            if (audioStopTimeout) clearTimeout(audioStopTimeout);
            audio.pause(); audio.currentTime = 0;

            document.getElementById('currentAudio').textContent = audioFile;
            document.getElementById('audioInfo').style.display = 'block';
            document.getElementById('choiceButtons').style.display = 'flex';
            if (document.getElementById('timer')) document.getElementById('timer').style.display = 'none';
            document.getElementById('leftBtn').disabled = true; // keep disabled until end
            document.getElementById('rightBtn').disabled = true;
            isWaitingForChoice = false;
            trialStartTime = Date.now();
            currentTrial = lastTrialNumber;

            audio.src = '/audio/' + encodeURIComponent(audioFile);
            audio.load();
            audio.autoplay = false; audio.muted = false; audio.pause(); audio.currentTime = 0;

            // 12秒流程：5s静默 + 5s音频 + 2s静默
            audioPlayTimeout = setTimeout(() => {
                if (!audioEnabled) return;
                audio.currentTime = 0; const p = audio.play(); if (p) p.catch(()=>{});
            }, 5000);

            audioStopTimeout = setTimeout(() => { audio.pause(); }, 10000);

            // 倒计时隐藏，仅在结束时提示
            let timeLeft = 12;
            const timerEl = document.getElementById('timer');
            if (timerEl) timerEl.style.display = 'none';
            const cd = setInterval(() => {
                timeLeft--;
                if (timeLeft <= 0) {
                    clearInterval(cd);
                    if (timerEl) { timerEl.style.display = 'block'; timerEl.textContent = '时间到！请做出选择'; }
                    document.getElementById('leftBtn').disabled = false;
                    document.getElementById('rightBtn').disabled = false;
                    isWaitingForChoice = true;
                }
            }, 1000);
        }

        function makeChoice(choice) {
            if (!isWaitingForChoice) return;
            const reactionTime = Date.now() - trialStartTime;
            const body = {
                username,
                participant_id: username,
                trial_number: currentTrial,
                audio_file: document.getElementById('currentAudio').textContent,
                choice,
                reaction_time: reactionTime
            };
            fetch('/participant_choice', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
                .then(r => r.json())
                .then(() => { document.getElementById('choiceButtons').style.display = 'none'; isWaitingForChoice = false; })
                .catch(()=>{});
        }

        document.addEventListener('keydown', function(event) {
            if (!isWaitingForChoice) return;
            if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') { makeChoice('left'); event.preventDefault(); }
            else if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') { makeChoice('right'); event.preventDefault(); }
        });
    </script>
</body>
</html>
    ''')


@app.route('/status')
def status():
    username = request.args.get('username', '').strip()
    state = init_user_if_needed(username)
    current_audio = ''
    total_trials = len(state['audio_order']) if state else 0
    if state and state['is_running'] and state['current_trial'] > 0 and state['current_trial'] <= total_trials:
        trial_index = state['current_trial'] - 1
        if 0 <= trial_index < total_trials:
            audio_index = state['audio_order'][trial_index]
            current_audio = audio_files[audio_index]
    return jsonify({
        'is_running': bool(state and state['is_running']),
        'current_trial': state['current_trial'] if state else 0,
        'total_trials': total_trials,
        'current_audio': current_audio
    })


@app.route('/audio/<filename>')
def serve_audio(filename):
    from flask import send_file
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path)
    return "Audio file not found", 404


@app.route('/video')
def serve_video():
    from flask import send_file
    if os.path.exists(VIDEO_PATH):
        return send_file(VIDEO_PATH)
    return "Video file not found", 404


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': '缺少用户名'}), 400
    state = init_user_if_needed(username)
    if state['is_running']:
        return jsonify({'error': '该用户的实验已在运行中'}), 400
    # 独立线程运行该用户的实验
    t = threading.Thread(target=run_experiment, args=(username,), daemon=True)
    with user_thread_locks:
        user_threads[username] = t
    t.start()
    return jsonify({'message': '实验已开始', 'username': username})


@app.route('/stop_experiment', methods=['POST'])
def stop_experiment():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': '缺少用户名'}), 400
    state = init_user_if_needed(username)
    state['is_running'] = False
    save_results(username)
    return jsonify({'message': '实验已停止', 'username': username})


@app.route('/reset_experiment', methods=['POST'])
def reset_experiment():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': '缺少用户名'}), 400
    # 重新初始化用户状态
    seed = int(time.time() * 1000) % 2**32
    rng = random.Random(seed)
    order = list(range(len(audio_files)))
    rng.shuffle(order)
    user_data[username] = {
        'trials': [],
        'current_trial': 0,
        'is_running': False,
        'trial_completed': {},
        'start_time': None,
        'end_time': None,
        'audio_order': order,
        'random_seed': seed
    }
    return jsonify({'message': '实验已重置', 'username': username})


@app.route('/participant_choice', methods=['POST'])
def participant_choice():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify({'error': '缺少用户名'}), 400
    state = init_user_if_needed(username)
    if not state:
        return jsonify({'error': '用户状态初始化失败'}), 400

    choice_data = {
        'participant_id': data.get('participant_id', username),
        'trial_number': data['trial_number'],
        'audio_file': data['audio_file'],
        'choice': data['choice'],
        'reaction_time': data['reaction_time'],
        'timestamp': datetime.now().isoformat()
    }
    state['trials'].append(choice_data)
    state['trial_completed'][choice_data['trial_number']] = True
    save_results(username)
    return jsonify({'message': '选择已记录', 'username': username})


if __name__ == '__main__':
    print("=" * 60)
    print("音频实验 Web 服务器（多用户版）")
    print("=" * 60)
    print(f"音频目录: {AUDIO_DIR}")
    print(f"视频文件: {VIDEO_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"发现音频文件: {len(audio_files)}")
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)


