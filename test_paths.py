import os

def test_file_paths():
    """测试文件路径是否存在"""
    print("=== 测试文件路径 ===")
    
    # 测试TXT文件
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
    ]
    
    print("TXT文件:")
    for i, f in enumerate(txt_files):
        exists = os.path.exists(f)
        print(f"  {i+1}. {os.path.basename(f)} - {'存在' if exists else '不存在'}")
    
    # 测试CSV文件
    csv_files = [
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai_1.csv",
        r"D:\D\research\audioset下载\experiment_output2\aiwenkai2.csv",
    ]
    
    print("\nCSV文件:")
    for i, f in enumerate(csv_files):
        exists = os.path.exists(f)
        print(f"  {i+1}. {os.path.basename(f)} - {'存在' if exists else '不存在'}")
    
    # 测试映射文件
    mapping_files = [
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder1\file_mapping_folder1.csv",
        r"D:\D\research\audioset下载\音频配对合成_新版\分组音频\folder2\file_mapping_folder2.csv",
    ]
    
    print("\n映射文件:")
    for i, f in enumerate(mapping_files):
        exists = os.path.exists(f)
        print(f"  {i+1}. {os.path.basename(f)} - {'存在' if exists else '不存在'}")

if __name__ == "__main__":
    test_file_paths()

