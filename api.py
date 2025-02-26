# -*- coding: utf-8 -*-
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
from io import BytesIO
classifier = pipeline("image-classification", model="wellCh4n/tomato-leaf-disease-classification-resnet50")

app = FastAPI()

name_dict = {
    'A tomato leaf with Tomato Yellow Leaf Curl Virus': '黄叶卷曲病毒',
    'A tomato leaf with Bacterial Spot': '细菌性斑点病',
    'A tomato leaf with Late Blight': '晚疫病',
    'A tomato leaf with Septoria Leaf Spot': '斑枯病',
    'A tomato leaf with Spider Mites Two-spotted Spider Mite': '蜘蛛螨',
    'A healthy tomato leaf': '健康',
    'A tomato leaf with Target Spot': '斑点病',
    'A tomato leaf with Early Blight': '早疫病',
    'A tomato leaf with Leaf Mold': '叶霉病',
    'A tomato leaf with Tomato Mosaic Virus': '番茄花叶病毒',
}

@app.post("/classification")
async def classification(file: Annotated[UploadFile, File()]):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    result = classifier(image)[0]

    label = result['label']
    name = name_dict.get(label)
    return {
        "classification_name": name,
        "classification": result['label'],
        "confidence": result['score']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="0.0.0.0", port=8000, reload=True)