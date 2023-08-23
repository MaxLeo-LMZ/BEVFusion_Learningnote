# 这段代码的目的是在递归地遍历一个 Python 对象（字典、列表），找到其中以 ${ 开头、以 } 结尾的字符串，将其计算为相应的值，并替换到对象中。
# 这个函数在某些情况下可以用于在配置文件中使用表达式，并在运行时计算表达式的值。
import copy

__all__ = ["recursive_eval"]


def recursive_eval(obj, globals=None):
    # 如果未提供 globals 参数，则将 globals 初始化为 obj 的深拷贝
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj
