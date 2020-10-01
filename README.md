# demo
我在写一段代码的时候遇到一点问题，希望得到帮助

下面是我的代码

# 我们选择的数据
t = [1,3,6,7,8,9,10,11,13,14,15,16,17,18,20,21,22,23,24,25]
mt = [1,9.10.15.49.77,115,124,156,208,277,323,373,405,
 457,508,550,590,617,638]


#这是一个函数
def logistic_increase_function(p, t):
    a, b = p
    return a * (1 - b * t + ((b * t) ** 2) / 2) * 1 / exp(t * b)


# 定义预测误差函数
def err_f(p, t, y):
    return logistic_increase_function(p, t) - y


# 参数初始值
logistic_p0 = [30, 0.3]

# 利用最小二乘法求解参数
logistic_params = leastsq(err_f, np.array(logistic_p0, dtype="float64"), args=(t, mt)

# 报错信息是这样的

"""
TypeError: Cannot cast array data from dtype('O') to dtype('float64') according to the rule 'safe'
Traceback (most recent call last):
  File "C:/Users/Administrator/PycharmProjects/untitled2/Demo/test13.py", line 57, in <module>
    logistic_params = leastsq(err_f, np.array(logistic_p0, dtype="float64"), args=(t, China_y))
  File "C:\Users\Administrator\PycharmProjects\untitled2\venv\lib\site-packages\scipy\optimize\minpack.py", line 422, in leastsq
    retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol,
minpack.error: Result from function call is not a proper array of floats.
"""
