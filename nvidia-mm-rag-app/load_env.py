import os
import base64
import fitz
from io import BytesIO
from PIL import Image
import requests
import logging
from llama_index.llms.nvidia import NVIDIA

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def set_environment_variables():
#     """Set necessary environment variables."""
#     nvidia_api_key = os.getenv("NVIDIA_API_KEY")


def get_b64_image_from_content(image_content):
    """Convert image content to base64 encoded string."""
    img = Image.open(BytesIO(image_content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_graph(image_content):
    """Determine if an image is a graph, plot, chart, or table."""
    res = describe_image(image_content)
    return any(keyword in res.lower() for keyword in ["graph", "plot", "chart", "table"])


# def process_graph(image_content):
#     """Process a graph image and generate a description."""
#     deplot_description = process_graph_deplot(image_content)
#     mixtral = NVIDIA(model_name="meta/llama-3.1-70b-instruct")
#     response = mixtral.complete("Your responsibility is to help students, especially those with learning differences such as dyslexia, understand and solve problems from any content they upload, including charts, data tables, or images of their homework. These may contain questions from subjects like English, mathematics, science, calculus, or any topic studied in primary or secondary school. Use simple, step-by-step explanations, breaking down complex information into easily digestible parts. Engage multiple senses by incorporating visual aids, verbal explanations, and, where applicable, assistive technologies like text-to-speech. Encourage problem-solving by providing hints rather than direct answers, using analogies and relatable examples to reinforce concepts. Highlight key points in the explanation, using bold or colored text where helpful. Always be mindful of the specific needs of dyslexic learners by keeping language clear and instructions structured, ensuring they understand the problem and gain confidence in finding solutions." + deplot_description)
#     return response.text

def process_graph(image_content):
    """Process a graph image and generate a description."""
    deplot_description = process_graph_deplot(image_content)
    # Use LLMChain with a persona/system prompt
    mixtral = NVIDIA(model_name="meta/llama-3.1-70b-instruct")

    # Define a template that includes the persona/system prompt
    template = """
    You are a highly supportive educational assistant. Your responsibility is to help students, especially those with learning differences such as dyslexia, understand and solve problems from any content they upload, including charts, data tables, or images of their homework. Use simple, step-by-step explanations, and provide hints rather than direct answers.
    Break down big and complicated words into smaller, easier-to-understand words. Use visual aids, verbal explanations, and, where applicable, assistive technologies like text-to-speech. Encourage problem-solving by providing hints rather than direct answers, using analogies and relatable examples to reinforce concepts. Highlight key points in the explanation, using bold or colored text where helpful. Always be mindful of the specific needs of dyslexic learners by keeping language clear and instructions structured, ensuring they understand the problem and gain confidence in finding solutions.
    
    Examples of Big Words:
    - "Photosynthesis" can be broken down into "photo" and "syn" "the" "sis"."
    - "Hippopotamus" can be broken down into "hippo" and "pot" "a" "mus".
    - "Metamorphosis" can be broken down into "meta" and "mor" "pho" "sis".
    - "Thermodynamics" can be broken down into "thermo" and "dy" "na" "mics".
    - "Electromagnetic" can be broken down into "electro" and "mag" "net" "ic".
    - "Photosynthesis" can be broken down into "photo" and "syn" "the" "sis".
    
    Here is the input content:
    {deplot_description}
    
    Please break down the words in the output as well so that students with dyslexia can read and understand better.
    """
    prompt = PromptTemplate(
        input_variables=["deplot_description"], template=template)

    # Set up the chain with the persona
    # chain = LLMChain(llm=mixtral, prompt=prompt)
    p = QueryPipeline(chain=[mixtral, prompt], verbose=True)
    response = p.run(deplot_description=deplot_description)

    return response


# Set up logging configuration if not already set
logging.basicConfig(level=logging.ERROR, filename='error_log.txt',
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def describe_image(image_content):
    """Generate a description of an image using NVIDIA API."""
    try:
        image_b64 = get_b64_image_from_content(image_content)
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")

        if not nvidia_api_key:
            raise ValueError(
                "NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Accept": "application/json"
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'You are an AI assistant that helps describe images in detail. <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "seed": 0,
            "stream": False
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        # Handle 401 Unauthorized
        if response.status_code == 401:
            logging.error(f"Authorization failed: {
                          response.status_code} - {response.text}")
            return "Authorization error: Please check the NVIDIA API key or your API permissions."

        # Raise for other HTTP errors
        response.raise_for_status()

        response_json = response.json()

        # Check the response structure
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]['message']['content']
        else:
            logging.error(f"Unexpected response structure: {response_json}")
            return "Unexpected response structure from the server."

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {
                      http_err} - Response: {response.text}")
        return "HTTP error occurred while processing the image."
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return "A request error occurred while processing the image."
    except ValueError as json_err:
        logging.error(f"JSON decode error: {
                      json_err} - Response: {response.text}")
        return "Failed to parse JSON response."
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."


def process_graph_deplot(image_content):
    """Process a graph image using NVIDIA's Deplot API."""
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
    image_b64 = get_b64_image_from_content(image_content)
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    if not nvidia_api_key:
        raise ValueError(
            "NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.20,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]['message']['content']


def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """Extract text above and below a given bounding box on a page."""
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(
            abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        horizontal_overlap = max(
            0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text


def process_text_blocks(text_blocks, char_count_threshold=500):
    """Group text blocks based on a character count threshold."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # Check if the block is of text type
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    temp_dir = os.path.join(os.getcwd(), "vectorstore",
                            "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    return temp_file_path
