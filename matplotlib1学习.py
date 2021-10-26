# 导入需要的包
import matplotlib.pyplot as plt
import numpy as np

# x坐标距离
x = np.linspace(-2*np.pi,2*np.pi,1000)  # 从-2pi到2pi画1000个点
# 转化为直角坐标系
plt.figure(figsize=(6,6))   # 设置画布的大小
# y = sin(x)
plt.plot(x,np.sin(x),label="y=sin(x)")
# y = cos(x)
plt.plot(x,np.cos(x),label="y=cos(x)")
plt.legend()
# 转化为直角坐标系
ax = plt.gca()      # 获取当前坐标轴
ax.spines['bottom'].set_position(('data', 0))       # 'data'表示按数值挪动，其后数字代表挪动到Y轴的刻度值
ax.spines['left'].set_position(('data', 0))         # 'data'表示按数值挪动，其后数字代表挪动到X轴的刻度值
ax.spines['top'].set_color('none')  # 设置顶部支柱的颜色为空
ax.spines['right'].set_color('none')  # 设置右边支柱的颜色为空
# 定义坐标箭头
ax.arrow(0, 0, 6.5, 0, head_width=0.2, head_length=0.5,color='black',linewidth=0)
ax.arrow(0, 0, 0, 5.5, head_width=0.2, head_length=0.5,color='black',linewidth=0)
# 自定义x和y坐标
ax.set_yticks([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,])
ax.set_xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
plt.show()      # 展示画布