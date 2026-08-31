import os

def parse_stereo_filename(filename):
    """
    解析立体声音频文件名，提取左右声道类别信息
    """
    try:
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]

        # 检查是否以 'nn_' 或 'rn_' 开头
        if not (name_without_ext.startswith('nn_') or name_without_ext.startswith('rn_')):
            return None, None

        # 移除 'nn_' 或 'rn_' 前缀
        if name_without_ext.startswith('nn_'):
            name_without_prefix = name_without_ext[3:]
        else:  # rn_
            name_without_prefix = name_without_ext[3:]

        # 按 '+' 分割左右声道
        if '+' not in name_without_prefix:
            return None, None

        left_part, right_part = name_without_prefix.split('+', 1)

        # 提取左声道类别（第一个下划线前的部分）
        left_underscore_pos = left_part.find('_')
        if left_underscore_pos == -1:
            return None, None
        left_category = left_part[:left_underscore_pos]  # 第一个下划线前的部分

        # 提取右声道类别（最后一个下划线前的部分）
        right_underscore_pos = right_part.rfind('_')  # 从右边开始找最后一个下划线
        if right_underscore_pos == -1:
            return None, None
        right_category = right_part[:right_underscore_pos]  # 最后一个下划线前的部分

        return left_category, right_category

    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None, None

def test_parser():
    """测试文件名解析"""
    test_filename = "nn_Pulse_vMq-4daVG1I+Whack, thwack_Ob9WB6eg1H0.wav"
    
    print(f"测试文件名: {test_filename}")
    
    left_category, right_category = parse_stereo_filename(test_filename)
    
    if left_category and right_category:
        print(f"解析成功!")
        print(f"左声道类别: {left_category}")
        print(f"右声道类别: {right_category}")
    else:
        print("解析失败!")

if __name__ == "__main__":
    test_parser()
