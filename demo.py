from pre_dataset import create_noise
from config import config

cf = config.get_config()
text = "mùa xuân sang có hoa anh đào , mùa xuân sang có hoa đào anh"
print(create_noise(cf, text))