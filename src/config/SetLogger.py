# -*- coding utf-8, LF -*-

import os
import sys
# from logging import WARNING, ERROR, CRITICAL, config
from datetime import datetime, timedelta
from logging import (DEBUG, FileHandler, Formatter, Logger, StreamHandler,
                     getLogger)

sys.path.append(os.getcwd() + "/src")
from config.params import LOG_DIR


def logger_conf() -> Logger:
    """
    ログ取得に関する関数

    Args:
        None

    Returns:
        Logger :

    """
    time = datetime.strftime(datetime.now() + timedelta(hours=9), "%Y%m%d%H%M%S%f")
    # ロガーの生成
    logger = getLogger("main")
    # 出力レベルの設定
    logger.setLevel(DEBUG)

    # ハンドラの生成
    sh = StreamHandler(sys.stdout)
    # 出力レベルの設定
    sh.setLevel(DEBUG)
    # sh.setLevel(WARNING)

    fh = FileHandler(filename=LOG_DIR+"/{time}.log".format(time=time), encoding='utf-8')
    fh.setLevel(DEBUG)

    # フォーマッタの生成（第一引数はメッセージのフォーマット文字列、第二引数は日付時刻のフォーマット文字列）
    fmt_terminal = Formatter("%(asctime)s :【 %(name)s 】%(message)s : %(addinfo)s\n", "%Y-%m-%d %H:%M:%S")
    fmt_file = Formatter("%(asctime)s 【 %(message)s 】%(addinfo)s\n", "%Y-%m-%d %H:%M:%S")

    # フォーマッタの登録
    sh.setFormatter(fmt_terminal)
    fh.setFormatter(fmt_file)

    # ハンドラの登録
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.debug("START", extra={"addinfo": "ログ取得開始"})
    return logger


if __name__ == "__main__":
    logger_a = logger_conf()
    logger = getLogger("res_root").getChild(__name__)
    logger.debug('test', extra={"addinfo": "test"})

