from PIL import Image

def crop_and_resize_image(
        input_path: str,
        output_path: str,
        crop_box: tuple,   # (left, upper, right, lower)
        scale_factor: int  # 정수배 (예: 2, 3, 4)
):
    # 이미지 열기
    image = Image.open(input_path).convert("RGB")

    # Crop 영역 지정
    cropped = image.crop(crop_box)  # (left, upper, right, lower)

    # 정수배 resize
    new_width = cropped.width * scale_factor
    new_height = cropped.height * scale_factor
    resized = cropped.resize((new_width, new_height), Image.BICUBIC)

    # 저장
    resized.save(output_path)
    print(f"Saved resized image to {output_path}")

# 🔧 사용 예시
crop_box = (60, 110, 410, 450)  # 왼쪽 위 (50,50) ~ 오른쪽 아래 (200,200)
scale_factor = 3
input_image_path = "data/캡처2.PNG"
output_image_path = "data/output_resized.jpg"

crop_and_resize_image(input_image_path, output_image_path, crop_box, scale_factor)
