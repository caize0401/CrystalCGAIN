import os
import heapq


def find_top_folders_with_most_files(root_dir, top_n=10):
    min_heap = []

    # 遍历所有文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        num_files = len(filenames)

        # 使用最小堆保持前 N 个文件夹
        if len(min_heap) < top_n:
            heapq.heappush(min_heap, (num_files, dirpath))
        else:
            heapq.heappushpop(min_heap, (num_files, dirpath))

    # 将堆中的内容转换为列表并排序
    top_folders = sorted(min_heap, key=lambda x: x[0], reverse=True)

    return top_folders


if __name__ == "__main__":
    root_directory = input("请输入要遍历的文件夹路径: ")
    top_folders = find_top_folders_with_most_files(root_directory, top_n=10)

    print("文件数量最多的前 10 个文件夹:")
    for count, folder in top_folders:
        print(f"{folder} - 文件数量: {count}")