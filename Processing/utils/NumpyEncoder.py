import numpy as np
import json
class NumpyEncoder(json.JSONEncoder):
    """NumPy 데이터 타입을 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

