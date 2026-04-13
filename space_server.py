import os

import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("api_deepseek:app", host="0.0.0.0", port=port)
