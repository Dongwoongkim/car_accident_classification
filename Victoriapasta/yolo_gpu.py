from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()

import torch

print(torch.cuda.is_available())  # True가 출력되어야 합니다.
print(torch.cuda.device_count())  # 1 이상의 숫자가 출력되어야 합니다.
print(torch.cuda.get_device_name(0))  # GPU 이름이 출력되어야 합니다.