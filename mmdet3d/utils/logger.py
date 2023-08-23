# 这段代码主要用于获取根日志记录器，并为其添加过滤器，以便根据关键字过滤处理日志记录。
# 它可用于在代码中设置日志记录的配置，如记录级别、输出位置等。
import logging
from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO, name="mmdet3d"):
    """Get root logger and add a keyword filter to it.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
        log_file：日志文件的路径，可选参数，默认为 None。
        log_level：日志的级别，可选参数，默认为 logging.INFO，即 INFO 级别。
        name：根日志记录器的名称，也作为一个过滤关键字，可选参数，默认为 'mmdet3d'。

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # add a logging filter
    # 根据指定的 name 进行过滤,检查记录中是否包含了指定的 name。如果包含了，就允许记录通过，否则不允许。
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1
    # 函数返回获取到的日志记录器对象 logger
    return logger
