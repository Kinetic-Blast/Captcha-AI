from captcha.image import ImageCaptcha
import random
from PIL import Image

characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ23456789'
captcha_text = ''.join(random.choice(characters) for _ in range(1))
size=(100, 100)

captcha = ImageCaptcha()
captcha_image_bytes = captcha.generate(captcha_text)
captcha_image = Image.open(captcha_image_bytes)
captcha_image = captcha_image.resize(size)

captcha_image.save("out.png")