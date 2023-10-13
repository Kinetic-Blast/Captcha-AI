import os
from captcha.image import ImageCaptcha
from PIL import Image

# Define the custom character set
characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ23456789'
num_images_per_character = 10000
output_directory = '\AI work\captcha_output'  # Change this to your desired output directory

def generate_captcha(char, size=(100, 100), length=1):
    captcha = ImageCaptcha()
    captcha_text = char
    captcha_image_bytes = captcha.generate(captcha_text)
    captcha_image = Image.open(captcha_image_bytes)
    captcha_image = captcha_image.resize(size)
    return captcha_image, captcha_text

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for char in characters:
    char_directory = os.path.join(output_directory, char)
    if not os.path.exists(char_directory):
        os.makedirs(char_directory)

    for i in range(num_images_per_character):
        captcha_image, captcha_text = generate_captcha(char)
        output_filename = f'{char}_{i}.png'
        output_path = os.path.join(char_directory, output_filename)

        captcha_image.save(output_path)
    print(f"Saved CAPTCHA image for '{char}'+' in '{char_directory}'")
    







