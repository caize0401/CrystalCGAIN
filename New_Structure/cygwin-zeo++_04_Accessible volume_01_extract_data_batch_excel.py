import os
import subprocess
import openpyxl

# 创建一个新的Excel工作簿
wb = openpyxl.Workbook()
ws = wb.active

# 添加表头
headers = ["文件名", "探针半径", "通道半径", "单位晶胞体积", "密度", "（可触及）单位晶胞体积", "（可触及）孔隙率", "（可触及）材料质量单位的体积",
           "（不可触及）单位晶胞体积", "（不可触及）孔隙率", "（不可触及）材料质量单位的体积",
           "通道数量", "通道可访问体积", "孔隙数量", "孔隙可访问体积"]
ws.append(headers)

# 参数设置
probe_radious = 0.05
chan_radius = 0.05

# 设置 Cygwin Bash Shell 和 zeo++ 命令的绝对路径
cygwin_bash = r'E:\materials_software\Software_Cygwin\bin\bash.exe'
zeo_path = r'/cygdrive/e/materials_software/Software_zeo++/zeo++-0.3/network'

# 设置要遍历的文件夹路径
folder_path = r'D:\python_program\GAIN\DETECTION\PCOD_cell_formula_units_z=8'
folder_cygwin_path = f'd/python_program/GAIN/DETECTION/PCOD_cell_formula_units_z=8'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.cif'):
        name = filename.split('.')[0]
        cif_path = "/cygdrive/" + folder_cygwin_path + "/" + filename
        zeo_command = f'{zeo_path} -ha -vol {probe_radious} {chan_radius} 50000 {cif_path}'

        # 使用 subprocess 运行 Cygwin Bash 并在其中执行 zeo++ 命令以获取通道维度
        try:
            subprocess.run(f'"{cygwin_bash}" --login -c "{zeo_command}"',
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # 读取文件
            filename_without_extension = os.path.splitext(filename)[0]
            vol_file = rf'D:\python_program\GAIN\DETECTION\PCOD_cell_formula_units_z=8\{filename_without_extension}.vol'

            # 读取并解析 vol 文件
            with open(vol_file, 'r') as file:
                lines = file.read().splitlines()

            # 提取单位晶胞体积和密度
            unitcell_volume = density = "N/A"
            try:
                unitcell_volume_line = lines[0]
                unitcell_volume = unitcell_volume_line.split()[3]
                density = unitcell_volume_line.split()[5]
            except IndexError:
                pass

            # 提取可访问体积（AV）的各项指标
            av_a3 = av_volume_fraction = av_cm3_g = "N/A"
            try:
                av_line = lines[0]
                av_a3 = av_line.split()[7]
                av_volume_fraction = av_line.split()[9]
                av_cm3_g = av_line.split()[11]
            except IndexError:
                pass

            # 提取不可访问体积（NAV）的各项指标
            nav_a3 = nav_volume_fraction = nav_cm3_g = "N/A"
            try:
                nav_line = lines[1]
                nav_a3 = nav_line.split()[7]
                nav_volume_fraction = nav_line.split()[9]
                nav_cm3_g = nav_line.split()[11]
            except IndexError:
                pass

            # 提取通道数量和通道体积
            number_of_channels = channel_volume = "N/A"
            try:
                channel_line = lines[2]
                number_of_channels = channel_line.split()[1]
                channel_volume = channel_line.split()[3]
            except IndexError:
                pass

            # 提取袋数量和袋体积
            number_of_pockets = pocket_volume = "N/A"
            try:
                pocket_line = lines[3]
                number_of_pockets = pocket_line.split()[1]
                pocket_volume = pocket_line.split()[3]
            except IndexError:
                pass

            # 写入Excel表格
            row_data = [filename, probe_radious, chan_radius, unitcell_volume, density, av_a3, av_volume_fraction, av_cm3_g,
                        nav_a3, nav_volume_fraction, nav_cm3_g, number_of_channels, channel_volume,
                        number_of_pockets, pocket_volume]
            ws.append(row_data)
            print(filename, "process finish")

        except subprocess.CalledProcessError as e:
            print("Error occurred:", e)
            print("Return code:", e.returncode)
            print("Output:", e.output)
        except IOError as e:
            print("Error reading file:", e)

# 保存Excel文件
wb.save(f"04_Accessible_volume_radius_p{probe_radious}_c{chan_radius}_PCOD_cell_formula_units_z=8.xlsx")