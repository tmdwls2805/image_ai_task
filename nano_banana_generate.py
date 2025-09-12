from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# 이미지 로드 (다른 이미지로 교체 가능)
model_image = Image.open('file/trump.jpg')
dress_image = Image.open('file/dress2.jpg')
background_image = Image.open('file/background2.jpg')

# JSON 기반 프롬프트 (범용적 설명)
prompt_data = {
    "person": {
        "description": (
            "From the attached images, use only one person: "
            "select the single individual who occupies the largest visible area in the photo. "
            "This person will be the main subject for the composition."
        ),
        "reference_image": f"{model_image}"
    },
    "clothing": {
        "description": (
            "From the clothing reference image, use only the outfit itself as guidance. "
            "Do not include the model wearing it, only extract the clothing design. "
            "This outfit must be applied to the selected person in the person image."
        ),
        "reference_image": f"{dress_image}"
    },
    "background": {
        "description": (
            "Use the background reference image as the environment. "
            "Do not simply paste the person on top; instead, blend the person naturally "
            "into the background so the final result looks seamless and realistic, "
            "as if photographed together. The result should be convincing enough "
            "to fool other viewers."
        ),
        "reference_image": f"{background_image}"
    },
    "style": {
        "lighting": "soft and natural",
        "tone": "fashion editorial, high detail, realistic fabric texture",
        "camera": "portrait shot, full body view",
        "pose": "smile and look at the camera and jump like a happy person"
    }
}


# JSON → 문자열
prompt = json.dumps(prompt_data, indent=2)

# 이미지 + 프롬프트 결합
response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[
        model_image,        # 사람
        dress_image,        # 옷
        background_image,   # 배경
        prompt              # JSON 설명
    ],
)

# 응답에서 이미지 추출
image_parts = [
    part.inline_data.data
    for part in response.candidates[0].content.parts
    if part.inline_data
]

# 저장 및 보기
if image_parts:
    image = Image.open(BytesIO(image_parts[0]))
    image.save('file/generated_image2.png')
    image.show()
