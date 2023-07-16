import numpy as np
from .difference import Difference

"""ニュートンラフソン法"""
class NewtonRaphson(object):
    """initialize"""
    def __init__(self, func):
        # 関数
        self.__func=func
        # 自動微分
        self.__ode=Difference()
        # 初期値x0
        self.__x0=np.random.randint(-10, 10)
        # Δxを設定
        self.__dx=0.01
        # λ学習係数
        self.__lambda=0.01
        # xの系列
        self.__x=[self.__x0]
        # 計算誤差
        self.__delta=1e-4

    """最適化"""
    def optimization(self, epoch=100):
        for i in range(epoch):
            self.__dx=-self.__func(self.__x[i])/self.__ode.forward(self.__func, self.__x[i], self.__dx)
            xi1=self.__x[i] + self.__dx
            self.__x.append(xi1)
            
            # 収束判定
            if np.abs(self.__x[i+1] - self.__x[i]) < self.__delta:
                print("収束しました。")
                return self.__x
        return self.__x