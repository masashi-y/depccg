# -*- coding: utf-8 -*-

from multiprocessing import Process, Pipe

"""
マルチスレッドで関数を実行するためのクラスです。
クラスの中から使えます。
"""
class MyPool:
    proc_num = 8

    def __init__(self, proc_num):
        self.proc_num = proc_num

    """
    指定した関数funcにargsの引数を一つ一つ与え実行します。
    これらはあらかじめ指定された数のプロセスで並列実行されます。
    """
    def map(self, func, args):
        def pipefunc(conn,arg):
            conn.send(func(arg))
            conn.close()
        ret = []
        k = 0
        while(k < len(args)):
            plist = []
            clist = []
            end = min(k + self.proc_num, len(args))
            for arg in args[k:end]:
                pconn, cconn = Pipe()
                plist.append(Process(target = pipefunc, args=(cconn,arg,)))
                clist.append(pconn)
            for p in plist:
                p.start()
            for conn in clist:
                ret.append(conn.recv())
            for p in plist:
                p.join()
            k += self.proc_num
        return ret
