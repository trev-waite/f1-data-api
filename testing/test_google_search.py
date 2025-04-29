import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

import asyncio

async def main():
    response = await client.aio.models.generate_content(
        model=model_id,
        contents="Can you give a few sentence recap of the Last F1 race in Jeddha?",
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )

    # for each in response.candidates[0].content.parts:
    #     print(each.text)
    print(response.text)

# Run the async function
asyncio.run(main())
