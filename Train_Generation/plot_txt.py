import matplotlib.pyplot as plt

# 定义一个函数来读取txt文件并提取epoch和porosity数据
def extract_data_from_txt(file_path):
    epochs = []
    porosity = []

    with open(file_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'porosity:' in line:
                # 提取epoch值
                epoch_value = int(line.split('Epoch')[1].split('/')[0].strip())
                epochs.append(epoch_value)

                # 提取porosity值
                porosity_value = float(line.split('porosity:')[1].split()[0].strip())
                porosity.append(porosity_value)

    return epochs, porosity

# 读取文件路径
file_path = 'model/gan_model_step_2/training_log.txt'  # 将这里替换为你的txt文件路径

# 提取数据
epochs, porosity = extract_data_from_txt(file_path)

# 画折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, porosity, linestyle='-', color='b', label='Porosity')

# 标注porosity: 0.0025
'''for i, p in enumerate(porosity):
    if p == 0.0025:
        plt.text(epochs[i], p, f"porosity: {p}", fontsize=12, color='red', ha='center')
        break'''

# 添加图表标题和标签
plt.title('Porosity Change Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Porosity', fontsize=12)

# 设置横坐标每隔50个epoch显示一次
xticks = list(range(0, max(epochs) + 1, 50))
plt.xticks(xticks)

plt.legend()

# 显示图表
plt.grid(True)
save_path = 'model/gan_model_step_2/porosity.png'
plt.savefig(save_path, format='png', bbox_inches='tight')
plt.show()
