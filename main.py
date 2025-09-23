from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from jigsawstack import JigsawStack
from gradio_client import Client, handle_file
from PIL import Image
import os, requests, joblib, base64, tempfile, json, pytesseract

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
JIGSAWSTACK_API_KEY = os.environ.get("JIGSAWSTACK_API_KEY")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
FACEBOOK_POSTS_SCRAPER_ACTOR_ID = os.environ.get("FACEBOOK_POSTS_SCRAPER_ACTOR_ID")
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")


## ML Model
jigsaw = JigsawStack(api_key=JIGSAWSTACK_API_KEY)

class NewsItem(BaseModel):
    text: str

def summarize_text(text: str) -> str:
    response = jigsaw.summary({
        "text": text,
        "type": "text",
        "max_characters": 100
    })
    return response["summary"]

@app.post("/news/predict")
def predict(request: NewsItem):
    input_text = request.text
    if len(input_text.split()) > 20:
        input_text = summarize_text(input_text)

    input_text = (
        "Is this statement true or false?"
        "If false, briefly explain why (e.g., lack of verified sources, contradictions with official PNG news, misinformation)."
        "Only use verified sources in Papua New Guinea, not social media.\n" + input_text)
    response = jigsaw.web.search({
        "query": input_text,
        "ai_overview": True,
        "safe_search": "moderate",
        "deep_research": True,
        "deep_research_config": {
            "max_depth": 3,
            "max_breadth": 3,
            "max_output_tokens": 100
        }
    })

    if 'false' in response["ai_overview"].lower():
        verdict = 'False'
    elif 'true' in response["ai_overview"].lower():
        verdict = 'True'
    else:
        verdict = 'Not sure'

    print(response['ai_overview'])
    reasoning = summarize_text(response['ai_overview'])
    if verdict == "False" and not any(kw in response['ai_overview'].lower() for kw in ["because", "according", "here's why"]):
        reasoning = f"The claim seems false due to lack of verified PNG sources.\nDetails: {reasoning}"

    sources = response.get("results", [])
    contradiction_notes = []
    for src in sources[:3]:  # only check top 3
        snippet = str(src.get("description", ""))
        url = src.get("url", "")
        if verdict == "False":
            if any(kw in snippet.lower() for kw in ["no evidence", "false", "denied", "contradicts", "unverified"]):
                contradiction_notes.append(f"Source ({url}) contradicts the claim: {snippet}")

    if contradiction_notes:
        reasoning += " | Supporting evidence: " + " ".join(contradiction_notes)
    
    result = f"{verdict}. {reasoning}"

    return {
        "explanation": result,
        "sources": response['results'][0]['url'] if response['results'][0]['url'] else "No sources found"
    }


## Article Scraper
from newspaper import Article
from apify_client import ApifyClient

class ArticleRequests(BaseModel):
    url: str


@app.post("/news/scrape")
def scrape_article(request: ArticleRequests):
    url = request.url
    if url.startswith("https://www.facebook.com/"):
        run_input = {
        "startUrls": [{ "url": url}],
        "resultsLimit": 1,
        "captionText": False,
        }

        client = ApifyClient(APIFY_API_TOKEN)

        run = client.actor(FACEBOOK_POSTS_SCRAPER_ACTOR_ID).call(run_input=run_input)
        text = ''
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            text += item.get('text', 'No text found') + "\n"

        text = summarize_text(text)

        return {"text": text}
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        print(url)
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return {"error": f"Failed to fetch the page: {response.status_code}"}
        
        html_content = response.text
        print("Fetched HTML length:", len(html_content))
        
        article = Article(url)
        article.set_html(html_content)
        article.parse()

        text = article.text
        text = summarize_text(text)

        return {"text": text}


## Text extraction from Image
class TextExtraction(BaseModel):
    image: str


@app.post("/news/textExtract")
def extract_text(request: TextExtraction):
    # client = Client("topdu/OpenOCR-Demo", hf_token=HF_TOKEN)
    try:
        image = request.image
        image_bytes = base64.b64decode(image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            temp_image.write(image_bytes)
            temp_image_path = temp_image.name

        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # For local development only
        text = pytesseract.image_to_string(Image.open(temp_image_path))

        os.remove(temp_image_path)

        return {"result": text}

        # result = client.predict(
        #     input_image=handle_file(temp_image_path),
        #     model_type_select="mobile",
        #     det_input_size_textbox=960,
        #     rec_drop_score=0.01,
        #     mask_thresh=0.3,
        #     box_thresh=0.6,
        #     unclip_ratio=1.5,
        #     det_score_mode="fast",
        #     api_name="/main"
        # )

        # ocr_json = result[0]
        # ocr_data = json.loads(ocr_json)
        # formated_text = " ".join(item['transcription'] for item in ocr_data)

        # return {"result": formated_text}
    except Exception as e:
        return {"error": f'Invalid base64 or some error with file handling: {str(e)}'}



## URL Authentication
# from urllib.parse import urlparse

# phish_model = open('./lr_model/phishing.pkl','rb')
# phish_model_ls = joblib.load(phish_model)

# class URLPrediction(BaseModel):
#     url:str

# def check_url(url: str) -> bool:
#     parsed = urlparse(url)
#     if not all([parsed.scheme, parsed.netloc]):
#         return False

#     try:
#         response = requests.head(url, allow_redirects=True, timeout=5)
#         return response.status_code < 400
#     except requests.RequestException:
#         return False

# @app.post("/url/predict")
# def url_predict(request: URLPrediction):
#     url = request.url
#     if not check_url(url):
#         return {"result": "Invalid URL"}
#     else:
#         predict = phish_model_ls.predict([url])
#         if predict == 'bad':
#             return {"result": "This is a phishing URL"}
#         else:
#             return {"result": "This is a legitimate URL"}


## Image Authentication
class ImageAuthentication(BaseModel):
    image: str  # URL or base64 string

@app.post("/image/analyze")
def image_predict(request: ImageAuthentication):
    client = Client("https://alsv-ai-genrated-image-detector.hf.space/", hf_token=HF_TOKEN)

    try:
        image = request.image
        image_bytes = base64.b64decode(image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            temp_image.write(image_bytes)
            temp_image_path = temp_image.name
        
        result = client.predict(
                        temp_image_path,
                        api_name="/predict",
        )
        os.remove(temp_image_path)
    except Exception as e:
        return {"error": f'Invalid base64 or some error with file handling: {str(e)}'}

    return {"result": result}