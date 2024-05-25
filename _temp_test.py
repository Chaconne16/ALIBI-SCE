from dataclasses import dataclass, field
import torch


def temp_test1():
    from tqdm import tqdm
    import time

    # 示例循环
    for i in tqdm(range(10), desc="Processing", unit="epoch"):
        time.sleep(0.1)


def temp_test2():
    @dataclass
    class Ts:
        arg1: int = 1
        arg2: str = 'test'

    ts1 = Ts()
    ts2 = Ts()
    ts1.arg1 += 1
    print(ts1.arg1, ts2.arg1, Ts.arg1)  # 不会修改类属性


def temp_test3():
    def adjust_learning_rate(epoch, lr):
        # if epoch < 30:  # warm-up
            # lr = lr * float(epoch + 1) / 30
        if epoch < 30 and epoch % 10 != 9:  # warm-up
            lr = lr  # --new lr adjusting trial
        elif epoch < 30 and epoch % 10 == 9:  # warm-up
            lr = lr / 3  # --new lr adjusting trial
        else:
            # lr = lr * (0.2 ** (epoch // 60))
            lr = lr * (0.98 ** (epoch // 60))  # --new lr adjusting trial
        return lr

    lr = 0.1
    for epc in range(200):
        lr = adjust_learning_rate(epc, lr)
        print(lr, end='; ')
        if epc % 10 == 9:
            print()


def temp_test4():
    def accuracy(ts1, ts2):
        return (ts1 == ts2).float().mean()

    ts3 = torch.arange(1, 10, dtype=torch.float)
    ts4 = torch.ones((9,))
    print(ts3, ts4)
    print(ts3 == ts4)
    print(accuracy(ts3, ts4))


if __name__ == '__main__':
    # temp_test1()
    # temp_test2()
    # temp_test3()
    temp_test4()
