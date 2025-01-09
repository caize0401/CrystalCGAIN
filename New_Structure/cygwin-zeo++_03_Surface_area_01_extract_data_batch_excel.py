import os
import subprocess
import openpyxl

# 创建一个新的Excel工作簿
wb = openpyxl.Workbook()
ws = wb.active

# 添加表头
headers = ["文件名", "探针半径", "通道半径", "单位晶胞体积", "密度", "（可触及）单位晶胞表面积", "（可触及）每体积（立方厘米）表面积（以平方米为单位）", "（可触及）每质量（克）材料的表面积",
           "（不可触及）单位晶胞表面积", "（不可触及）每体积（立方厘米）表面积（以平方米为单位）", "（不可触及）每质量（克）材料的表面积",
           "通道数量", "通道表面积", "袋数量", "袋表面积"]
ws.append(headers)

# 参数设置
probe_radious = 0.05
chan_radius = 0.05

# 设置 Cygwin Bash Shell 和 zeo++ 命令的绝对路径
cygwin_bash = r'E:\materials_software\Software_Cygwin\bin\bash.exe'
zeo_path = r'/cygdrive/e/materials_software/Software_zeo++/zeo++-0.3/network'

# 设置要遍历的文件夹路径
folder_path = r'D:\python_program\GAIN\DETECTION\new_structures_metastable_calculate_file'
folder_cygwin_path = f'd/python_program/GAIN/DETECTION/new_structures_metastable_calculate_file'
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.cif'):
        name = filename.split('.')[0]
        cif_path = "/cygdrive/" + folder_cygwin_path + "/" + filename
        zeo_command = f'{zeo_path} -ha -sa {probe_radious} {chan_radius} 2000 {cif_path}'

        try:
            subprocess.run(f'"{cygwin_bash}" --login -c "{zeo_command}"',
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # 读取文件
            filename_without_extension = os.path.splitext(filename)[0]
            sa_file = rf'D:\python_program\GAIN\DETECTION\new_structures_metastable_calculate_file\{filename_without_extension}.sa'

            # 读取并解析文件内容
            with open(sa_file, 'r') as file:
                lines = file.read().splitlines()

            # 提取单位晶胞体积和密度
            unitcell_volume = density = "N/A"  # 默认值为 "N/A"
            try:
                unitcell_volume_line = lines[0]
                unitcell_volume = unitcell_volume_line.split()[3]
                density = unitcell_volume_line.split()[5]
            except IndexError:
                pass  # 如果缺失值导致索引错误，则跳过

            # 提取可触及表面积（ASA）的各项指标
            asa_a2 = asa_m2_cm3 = asa_m2_g = "N/A"  # 默认值为 "N/A"
            try:
                asa_line = lines[0]
                asa_a2 = asa_line.split()[7]
                asa_m2_cm3 = asa_line.split()[9]
                asa_m2_g = asa_line.split()[11]
            except IndexError:
                pass

            # 提取不可触及表面积（NASA）的各项指标
            nasa_a2 = nasa_m2_cm3 = nasa_m2_g = "N/A"  # 默认值为 "N/A"
            try:
                nasa_line = lines[0]
                nasa_a2 = nasa_line.split()[13]
                nasa_m2_cm3 = nasa_line.split()[15]
                nasa_m2_g = nasa_line.split()[17]
            except IndexError:
                pass

            # 提取通道数量和通道表面积
            number_of_channels = channel_surface_area = "N/A"  # 默认值为 "N/A"
            try:
                channel_line = lines[1]
                number_of_channels = channel_line.split()[1]
                channel_surface_area = channel_line.split()[3]
            except IndexError:
                pass

            # 提取袋数量和袋表面积
            number_of_pockets = pocket_surface_area = "N/A"  # 默认值为 "N/A"
            try:
                pocket_line = lines[2]
                number_of_pockets = pocket_line.split()[1]
                pocket_surface_area = pocket_line.split()[3]
            except IndexError:
                pass

            # 写入Excel表格
            row_data = [filename, probe_radious, chan_radius, unitcell_volume, density, asa_a2, asa_m2_cm3, asa_m2_g,
                        nasa_a2, nasa_m2_cm3, nasa_m2_g, number_of_channels, channel_surface_area,
                        number_of_pockets, pocket_surface_area]
            ws.append(row_data)

            print(filename,"process finish")

        except subprocess.CalledProcessError as e:
            print("Error occurred:", e)
            print("Return code:", e.returncode)
            print("Output:", e.output)
        except IOError as e:
            print("Error reading file:", e)

# 保存Excel文件
wb.save(f"03_Surface_area_radius_p{probe_radious}_c{chan_radius}_83_new_structure_metastable.xlsx")
