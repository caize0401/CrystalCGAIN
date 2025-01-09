import os
import subprocess
import re
import pandas as pd

# 设置 Cygwin Bash Shell 和 zeo++ 命令的绝对路径
cygwin_bash = r'E:\materials_software\Software_Cygwin\bin\bash.exe'
zeo_path = r'/cygdrive/e/materials_software/Software_zeo++/zeo++-0.3/network'

# 设置要遍历的文件夹路径
folder_path = r'D:\python_program\GAIN\DETECTION\new_structures_stable_calculate_file'
folder_cygwin_path = f'd/python_program/GAIN/DETECTION/new_structures_stable_calculate_file'
# 设置探头半径范围和步长
probe_radii = [round(i * 0.1, 1) for i in range(1, 16)]  # 从0.1到1.5每隔0.1
data = {}

# 遍历文件夹中的所有文件和探头半径
for filename in os.listdir(folder_path):
    if filename.endswith('.cif'):
        name = filename.split('.')[0]
        cif_path = "/cygdrive/" + folder_cygwin_path + "/" + filename
        for chan in probe_radii:
            zeo_command = f'{zeo_path} -chan {chan} {cif_path}'
            try:
                # 使用 subprocess 运行 Cygwin Bash 并在其中执行 zeo++ 命令以获取通道维度
                result = subprocess.run(f'"{cygwin_bash}" --login -c "{zeo_command}"',
                                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

                # 使用正则表达式匹配所需的字符串
                if result.stdout:
                    match = re.search(r'Identified (\d+) channels and \d+ pockets.', result.stdout)
                    if match:
                        if filename not in data:
                            data[filename] = {}
                        data[filename][chan] = match.group(1)
                    else:
                        if filename not in data:
                            data[filename] = {}
                        data[filename][chan] = 0
                else:
                    if filename not in data:
                        data[filename] = {}
                    data[filename][chan] = 0

            except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print("Return code:", e.returncode)
                print("Output:", e.output)
            except IOError as e:
                print("Error reading file:", e)

            print(filename,"process finish")

# 将数据转换为DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# 将结果保存到Excel文件中
output_excel = '02_Channel_identification_and_dimensionality_33_new_structure_stable.xlsx'
df.to_excel(output_excel)
print(f"结果已保存到 {output_excel}")
