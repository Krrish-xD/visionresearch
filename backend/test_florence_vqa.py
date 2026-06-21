import asyncio
from PIL import Image
import os
import sys

# Add backend to path
sys.path.append("/home/kxd/Coding/visionresearch/backend")

from modules.caption import SceneCaptioner

async def main():
    print("Initializing...")
    captioner = SceneCaptioner()
    await captioner.load_model(device="cuda")
    
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color='red')
    
    print("Testing original ask_question...")
    answer = await captioner.ask_question(img, "what is in this image")
    print(f"Original Answer: '{answer}'")

if __name__ == "__main__":
    asyncio.run(main())
