#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试行为分析代码的核心逻辑
"""

class TestBehavioralLogic:
    """测试行为分析逻辑"""
    
    def __init__(self):
        """初始化测试"""
        # 定义主类别映射
        self.main_categories = {
            'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
            'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
            'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
            'music': ['Bass drum', 'Funny music', 'Sad music']
        }
        
        # 定义目标类别
        self.target_categories = {
            'main': {
                'High Ecology vs Low Ecology': 'High Ecology',
                'High Ecology vs speech': 'High Ecology',
                'High Ecology vs music': 'High Ecology',
                'Low Ecology vs speech': 'speech',
                'Low Ecology vs music': 'music',
                'speech vs music': 'speech'
            },
            'sub': {
                'Baby cry, infant cry vs Telephone bell ringing': 'Baby cry, infant cry',
                'Helicopter vs Computer keyboard': 'Helicopter',
                'Male speech, man speaking vs Female speech, woman speaking': 'Male speech, man speaking',
                'Bass drum vs Sad music': 'Bass drum'
            }
        }

    def get_main_category(self, sub_category):
        """根据子类别获取主类别"""
        for main_cat, sub_cats in self.main_categories.items():
            if sub_category in sub_cats:
                return main_cat
        return None

    def parse_audio_filename(self, filename):
        """解析音频文件名"""
        try:
            name_without_ext = filename.replace('.wav', '')
            
            if name_without_ext.startswith('nn_main_'):
                experiment_type = 'nn_main'
                name_part = name_without_ext[8:]
            elif name_without_ext.startswith('main_'):
                experiment_type = 'main'
                name_part = name_without_ext[5:]
            elif name_without_ext.startswith('sub_'):
                experiment_type = 'sub'
                name_part = name_without_ext[4:]
            else:
                return None, None, None
            
            if '+' not in name_part:
                return None, None, None
            
            left_part, right_part = name_part.split('+', 1)
            
            left_underscore_pos = left_part.find('_')
            if left_underscore_pos == -1:
                return None, None, None
            left_category = left_part[:left_underscore_pos]
            
            right_underscore_pos = right_part.find('_')
            if right_underscore_pos == -1:
                return None, None, None
            right_category = right_part[:right_underscore_pos]
            
            return experiment_type, left_category, right_category
            
        except Exception as e:
            print(f"解析文件名失败 {filename}: {e}")
            return None, None, None

    def determine_contrast_type(self, experiment_type, left_category, right_category):
        """确定对比类型"""
        if experiment_type in ['main', 'nn_main']:
            left_main = self.get_main_category(left_category)
            right_main = self.get_main_category(right_category)
            
            if left_main == right_main:
                return f"{left_main} vs {left_main}"
            else:
                return f"{left_main} vs {right_main}"
        
        elif experiment_type == 'sub':
            return f"{left_category} vs {right_category}"
        
        return None

    def is_target_selected(self, contrast_type, experiment_type, left_category, right_category, response):
        """判断是否选择了目标类别"""
        try:
            if experiment_type in ['main', 'nn_main']:
                target_main = self.target_categories['main'].get(contrast_type)
                if target_main is None:
                    return False
                
                left_main = self.get_main_category(left_category)
                right_main = self.get_main_category(right_category)
                
                if left_main == target_main:
                    return response == 1
                elif right_main == target_main:
                    return response == 2
                    
            elif experiment_type == 'sub':
                target_sub = self.target_categories['sub'].get(contrast_type)
                if target_sub is None:
                    return False
                
                if left_category == target_sub:
                    return response == 1
                elif right_category == target_sub:
                    return response == 2
            
            return False
            
        except Exception as e:
            print(f"判断目标选择失败: {e}")
            return False


def test_logic():
    """测试核心逻辑"""
    print("=== 测试行为分析逻辑 ===")
    
    tester = TestBehavioralLogic()
    
    # 测试用例1: main类型
    test_filename1 = 'main_Helicopter_Wr44Q8MQHL0_1+Computer keyboard_VQbC7Oth7wQ_1.wav'
    exp_type1, left_cat1, right_cat1 = tester.parse_audio_filename(test_filename1)
    print(f"测试1 - 文件名解析: {exp_type1}, {left_cat1}, {right_cat1}")
    
    contrast_type1 = tester.determine_contrast_type(exp_type1, left_cat1, right_cat1)
    print(f"测试1 - 对比类型: {contrast_type1}")
    
    is_target1 = tester.is_target_selected(contrast_type1, exp_type1, left_cat1, right_cat1, 1)
    print(f"测试1 - 选择左声道是否为目标: {is_target1}")
    
    # 测试用例2: sub类型
    test_filename2 = 'sub_Baby cry, infant cry_oPzAaB7LoqU_2+Telephone bell ringing_LJH4TFzWiW0_1.wav'
    exp_type2, left_cat2, right_cat2 = tester.parse_audio_filename(test_filename2)
    print(f"测试2 - 文件名解析: {exp_type2}, {left_cat2}, {right_cat2}")
    
    contrast_type2 = tester.determine_contrast_type(exp_type2, left_cat2, right_cat2)
    print(f"测试2 - 对比类型: {contrast_type2}")
    
    is_target2 = tester.is_target_selected(contrast_type2, exp_type2, left_cat2, right_cat2, 1)
    print(f"测试2 - 选择左声道是否为目标: {is_target2}")
    
    # 测试用例3: nn_main类型
    test_filename3 = 'nn_main_Helicopter_W3u2hj1x7gY_2+Telephone bell ringing__xuq9rBndUE_1.wav'
    exp_type3, left_cat3, right_cat3 = tester.parse_audio_filename(test_filename3)
    print(f"测试3 - 文件名解析: {exp_type3}, {left_cat3}, {right_cat3}")
    
    contrast_type3 = tester.determine_contrast_type(exp_type3, left_cat3, right_cat3)
    print(f"测试3 - 对比类型: {contrast_type3}")
    
    is_target3 = tester.is_target_selected(contrast_type3, exp_type3, left_cat3, right_cat3, 2)
    print(f"测试3 - 选择右声道是否为目标: {is_target3}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_logic()
