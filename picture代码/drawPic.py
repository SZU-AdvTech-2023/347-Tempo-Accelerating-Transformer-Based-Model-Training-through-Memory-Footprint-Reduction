import matplotlib.pyplot as plt
import re
import numpy as np
# 设置全局中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为你下载的中文字体名称
plt.rcParams['axes.unicode_minus'] = False


# 定义一个简单的平均滤波函数
def smooth_curve(data, window_size):
    box = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, box, mode='same')
    return smoothed_data


def analyzeMemory(fileName):
    memory_l, memory_r = [], []
    pattern_memory = r'\b(\d+\.\d+)MB\b'
    pattern_memory_rate = r'Usage (\d+\.\d+)%'
    with open(fileName, 'r') as file:
        line = file.readline()
        while line:
            memory_l.append(re.findall(pattern_memory, line)[0])
            memory_r.append(re.findall(pattern_memory_rate, line)[0])
            line = file.readline()
    memory_l = [round(float(i), 2) for i in memory_l]
    memory_r = [round(float(i), 2) for i in memory_r]
    return memory_l, memory_r


def analyzeAccuracy(fileName):
    lines = []

    with open(fileName, 'r') as file:
        log_string = file.readline()

        while log_string:
            # 使用正则表达式提取损失和准确率
            iter_match = re.findall(r'Iter:\s+(\d+),', log_string)
            train_loss_match = re.findall(r'Train Loss:\s+(\d+\.\d+),', log_string)
            train_acc_match = re.findall(r'Train Acc:\s+(\d+\.\d+)%', log_string)
            val_loss_match = re.findall(r'Val Loss:\s+(\d+\.\d+),', log_string)
            val_acc_match = re.findall(r'Val Acc:\s+(\d+\.\d+)%', log_string)

            # 如果匹配到结果则添加到列表
            if iter_match and train_loss_match and train_acc_match and val_loss_match and val_acc_match:
                iter_num = int(iter_match[0])
                train_loss = float(train_loss_match[0])
                train_acc = float(train_acc_match[0])
                val_loss = float(val_loss_match[0])
                val_acc = float(val_acc_match[0])

                lines.append({"iter_num": iter_num, "train_loss": train_loss,
                              "train_acc": train_acc, "val_loss": val_loss,
                              "val_acc": val_acc})

            # 读取下一行
            log_string = file.readline()

    return lines

"""
    第一条曲线，GPU占比
"""
memory_used_raw, memory_rate_raw = analyzeMemory("gpu_memory_log/gpu_memory_log.txt")
memory_used_tempo, memory_rate_tempo = analyzeMemory("gpu_memory_log/tempo_gpu_memory_log.txt")

# 设置窗口大小
window_size = 2
# 对memory_l数据进行平滑处理
smoothed_memory_raw = smooth_curve(memory_rate_raw, window_size)
smoothed_memory_tempo= smooth_curve(memory_rate_tempo, window_size)

# 绘制平滑后的曲线
X = np.arange(0, len(smoothed_memory_raw))
# plt.xlim(0, X[-1]+100)

plt.plot(X, smoothed_memory_raw, label=f'Smoothed Memory Usage (Window Size: {window_size})', linewidth=2, color='#3cb44b')

X = np.arange(0, len(smoothed_memory_tempo))
plt.plot(X, smoothed_memory_tempo, label=f'Smoothed Memory Usage (Window Size: {window_size})', linewidth=2, color='#4363d8')

plt.ylabel("GPU使用率(%)", fontsize=13)
plt.xlabel("时间轴", fontsize=13)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(["baseline", "tempo"], fontsize='large')
plt.savefig("gpu_rate.png")
plt.show()
plt.close()

"""
    训练指标：
"""
tempo = analyzeAccuracy("gpu_memory_log/tempo_train_log.txt")
baseline = analyzeAccuracy("gpu_memory_log/train_log.txt")
"""
    第二条曲线，训练集准确率
"""
X = [i['iter_num'] / 50 for i in tempo]
# plt.xlim(0, X[-1]+10)

Y_train_acc = [i['train_acc'] for i in tempo]
window_size = 2
Y_train_acc = smooth_curve(Y_train_acc, window_size=window_size)
plt.plot(X, Y_train_acc, linewidth=2, color='#3cb44b')

X = [i['iter_num'] / 50 for i in baseline]
Y2_train_acc = [i['train_acc'] for i in baseline]
Y2_train_acc = smooth_curve(Y2_train_acc, window_size=window_size)
plt.plot(X, Y2_train_acc, linewidth=2, color='#4363d8')
plt.legend(['tempo', 'baseline'], fontsize='large')
plt.xlabel("训练轮次(每50个batch)", fontsize=13)
plt.ylabel("准确率(%)", fontsize=13)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.savefig("train_accuracy.png")
plt.show()
plt.close()

"""
    第三条曲线，测试集准确率
"""
X = [i['iter_num'] / 50 for i in tempo]
Y_val_acc = [i['val_acc'] for i in tempo]
window_size = 1
Y_val_acc = smooth_curve(Y_val_acc, window_size=window_size)
plt.plot(X, Y_val_acc, linewidth=2, color='#3cb44b')

X = [i['iter_num'] / 50 for i in baseline]
Y2_val_acc = [i['val_acc'] for i in baseline]
Y2_val_acc = smooth_curve(Y2_val_acc, window_size=window_size)
plt.plot(X, Y2_val_acc, linewidth=2, color='#4363d8')
plt.legend(['tempo', 'baseline'], fontsize='large')
plt.xlabel("轮次(每50个batch)",fontsize=13)
plt.ylabel("准确率(%)", fontsize=13)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.savefig("test_accuracy.png")
plt.show()
plt.close()

"""
    第四条曲线，训练集loss
"""
X = [i['iter_num'] / 50 for i in tempo]
# plt.xlim(0, X[-1]+1)

Y_train_loss = [i['train_loss'] for i in tempo]
window_size = 2
# Y_train_loss = smooth_curve(Y_train_loss, window_size=window_size)
plt.plot(X, Y_train_loss, linewidth=2, color='#3cb44b')

X = [i['iter_num'] / 50 for i in baseline]
Y2_train_loss = [i['train_loss'] for i in baseline]
# Y2_train_loss= smooth_curve(Y2_train_loss, window_size=window_size)
plt.plot(X, Y2_train_loss, linewidth=2, color='#4363d8')
plt.legend(['tempo', 'baseline'], fontsize='large')
plt.xlabel("轮次(每50个batch)", fontsize=13)
plt.ylabel("损失", fontsize=13)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.savefig("train_loss.png")
plt.show()
plt.close()

"""
    第四条曲线，测试集loss
"""
# plt.xlim(0, X[-1]+1)

X = [i['iter_num'] / 50 for i in tempo]
Y_val_loss = [i['val_loss'] for i in tempo]
window_size = 2
# Y_val_loss = smooth_curve(Y_val_loss, window_size=window_size)
plt.plot(X, Y_val_loss, linewidth=2, color='#3cb44b')
X = [i['iter_num'] / 50 for i in baseline]
Y2_val_loss = [i['val_loss'] for i in baseline]
# Y2_val_loss= smooth_curve(Y2_val_loss, window_size=window_size)
plt.plot(X, Y2_val_loss, linewidth=2, color='#4363d8')
plt.legend(['tempo', 'baseline'], fontsize='large')
plt.xlabel("轮次(每50个batch)", fontsize=13)
plt.ylabel("损失", fontsize=13)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.savefig("test_loss.png")
plt.show()
plt.close()