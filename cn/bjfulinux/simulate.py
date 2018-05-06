import SingleRun
import random

times = 2000

print('Start simulate ...')
for i in range(times):
    random_thre = round(2 * random.random(), 2)  # 随机噪声上限
    if i % (times // 100) == 0:
        print('\r' + str(i // (times // 100)) + '%', end='')
    SingleRun.run_simulation(5, 3, random_thre)
print("\r100%\nComplete.")