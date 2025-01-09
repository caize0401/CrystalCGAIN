import matplotlib.pyplot as plt

# 定义函数读取文件并提取 RMSE 值
def read_rmse_from_txt(file_path):
    train_miss_rmse = []
    train_obse_rmse = []
    test_miss_rmse = []
    test_obse_rmse = []

    # 打开文件读取
    with open(file_path, 'r') as f:
        for line in f:
            # 解析 RMSE 值
            if "Train miss rmse" in line:
                parts = line.split(",")
                train_miss_rmse.append(float(parts[1].split(":")[1]))
                train_obse_rmse.append(float(parts[2].split(":")[1]))
                test_miss_rmse.append(float(parts[3].split(":")[1]))
                test_obse_rmse.append(float(parts[4].split(":")[1]))

    return train_miss_rmse, train_obse_rmse, test_miss_rmse, test_obse_rmse

# 从文件读取 RMSE 数据
file_path = "reconstruction.txt"  # 替换为你的txt文件路径
train_miss_rmse, train_obse_rmse, test_miss_rmse, test_obse_rmse = read_rmse_from_txt(file_path)
print(len(train_miss_rmse))
# 画出 RMSE 曲线
epochs = range(1, len(train_miss_rmse) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_miss_rmse, label='Train Miss RMSE', linewidth=1)
plt.plot(epochs, train_obse_rmse, label='Train Obse RMSE', linewidth=1)
plt.plot(epochs, test_miss_rmse, label='Test Miss RMSE', linewidth=1)
plt.plot(epochs, test_obse_rmse, label='Test Obse RMSE', linewidth=1)


# 添加标题和标签
plt.title('RMSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

# 显示图像
plt.grid(True)
plt.tight_layout()
# Save the plot as an image file
plt.savefig("reconstruction.png", dpi=500, bbox_inches="tight")
plt.show()
