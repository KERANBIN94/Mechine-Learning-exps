# -*- coding: utf-8 -*-
"""
将指定文件转换为 big5 编码。
"""

import os

def convert_files_to_big5(files_to_convert, source_encoding='utf-8', target_encoding='big5'):
    """
    读取文件列表，并将其从源编码转换为目标编码。

    Args:
        files_to_convert (list): 文件路径列表。
        source_encoding (str): 文件的原始编码。
        target_encoding (str): 要转换的目标编码。
    """
    print(f"开始转换文件到 {target_encoding} 编码...")
    
    for filename in files_to_convert:
        if not os.path.exists(filename):
            print(f"警告: 文件 '{filename}' 不存在，已跳过。")
            continue

        try:
            # 1. 使用源编码读取文件内容
            print(f"正在读取 '{filename}' (编码: {source_encoding})...")
            with open(filename, 'r', encoding=source_encoding, errors='replace') as f:
                content = f.read()
            
            # 2. 使用目标编码写回文件（覆盖原文件）
            #    errors='replace' 会将无法在 big5 中表示的字符替换为 '?'
            print(f"正在写入 '{filename}' (编码: {target_encoding})...")
            with open(filename, 'w', encoding=target_encoding, errors='replace') as f:
                f.write(content)
                
            print(f"成功: 文件 '{filename}' 已转换为 {target_encoding} 编码。")

        except UnicodeDecodeError:
            print(f"错误: 使用 {source_encoding} 解码 '{filename}' 失败。")
            print("      请确认文件的实际编码是否正确，或尝试更改 source_encoding 参数。")
        except Exception as e:
            print(f"错误: 处理 '{filename}' 时发生未知错误: {e}")
        
        print("-" * 20)

if __name__ == "__main__":
    # 定义需要转换编码的文件列表
    csv_files = ['train.csv', 'test.csv']
    
    convert_files_to_big5(csv_files)
    
    print("所有文件处理完毕。")
