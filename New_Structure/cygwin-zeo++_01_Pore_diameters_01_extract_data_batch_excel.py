import os
import subprocess
import pandas as pd

# 设置 Cygwin Bash Shell 和 zeo++ 命令的绝对路径
cygwin_bash = r'E:\materials_software\Software_Cygwin\bin\bash.exe'
zeo_path = r'/cygdrive/e/materials_software/Software_zeo++/zeo++-0.3/network'

# 设置要遍历的文件夹路径
folder_path = r'D:\python_program\GAIN\DETECTION\new_structures_metastable_calculate_file'
folder_cygwin_path = f'd/python_program/GAIN/DETECTION/new_structures_metastable_calculate_file'

# 创建一个空的DataFrame来保存结果
data = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.cif'):
        name = filename.split('.')[0]

        cif_path = "/cygdrive/" + folder_cygwin_path + "/" + filename
        zeo_command = f'{zeo_path} -ha -res {cif_path}'

        try:
            # 使用 subprocess 运行 Cygwin Bash 并在其中执行 zeo++ 命令
            result = subprocess.run(f'"{cygwin_bash}" --login -c "{zeo_command}"',
                                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # 读取并解析 .res 文件
            res_file = os.path.join(folder_path, f'{name}.res')
            with open(res_file, 'r') as file:
                line = file.readline().strip()
                diameters = line.split()
                if len(diameters) == 4:  # 包含路径和三个直径值
                    data.append({
                        '文件名': filename,
                        '最大包含球体直径': diameters[1],
                        '自由球体直径': diameters[2],
                        '沿自由球体路径的包含球体直径': diameters[3]
                    })
            print(filename, "process finish")

        except subprocess.CalledProcessError as e:
            print("Error occurred for file:", filename)
            print("Error:", e)

# 将结果保存到Excel文件中
df = pd.DataFrame(data)
output_excel = '01_Pore_diameters_83_new_structure_metastable.xlsx'
df.to_excel(output_excel, index=False)
print(f"结果已保存到 {output_excel}")
