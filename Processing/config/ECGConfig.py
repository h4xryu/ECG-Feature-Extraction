from Processing.base import *


@dataclass
class ECGConfig:
    fs: float
    p_wave: bool = True
    u_wave: bool = False
    hermite_nums: List[int] = None
    rules: List[float] = None
    pre_r: Optional[int] = None  # Added pre_r attribute

    def __post_init__(self):
        if self.hermite_nums is None:
            if self.p_wave and self.u_wave:
                self.hermite_nums = [7, 6, 4, 2]  # [QRS, T, P, U]
            elif self.p_wave:
                self.hermite_nums = [7, 6, 4]  # [QRS, T, P]
            elif self.u_wave:
                self.hermite_nums = [7, 6, 2]  # [QRS, T, U]
            else:
                self.hermite_nums = [7, 6]  # [QRS, T]

        if self.rules is None:
            self.rules = [3, 3, 3, 3]  # Default optimization constraints

        # Initialize pre_r if not provided
        if self.pre_r is None:
            self.pre_r = None  # Will be calculated in prepare_data method