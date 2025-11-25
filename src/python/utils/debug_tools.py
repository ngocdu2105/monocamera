import inspect
from functools import wraps


def log_variables(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gọi hàm chính
            result = func(*args, **kwargs)

            # Lấy frame thông tin nơi hàm được gọi
            frame = inspect.currentframe().f_back
            info = inspect.getframeinfo(frame)

            # Lấy các biến cục bộ tại thời điểm gọi func (sau khi chạy)
            local_vars = frame.f_locals

            for var_name, value in local_vars.items():
                if not var_name.startswith("__") and var_name not in ["args", "kwargs"]:
                    logger.info(
                        f"file={info.filename} line={info.lineno} "
                        f"func={func.__name__} var={var_name} value={repr(value)}"
                    )

            return result

        return wrapper

    return decorator
