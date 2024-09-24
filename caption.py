import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


load_dotenv()

key = os.getenv("VISION_KEY")
endpoint = os.getenv("VISION_END_URL")


client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key=key))

test_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"

result = client.analyze_from_url(
    image_url=test_url, 
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS],
    gender_neutral_caption=True
)


if result.caption is not None:
    print(f" {result.caption.text},  confidence : {result.caption.confidence}")

print("------------------------------------------------------------------")

if result.dense_captions is not None:
    for caption in result.dense_captions.list:
        print(f" {caption.text}  confidence : {caption.confidence}")

