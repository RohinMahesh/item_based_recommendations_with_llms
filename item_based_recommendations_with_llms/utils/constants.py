import re

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768
EMBED_DIM = 1536
HTML_PATTERN = re.compile(r"<.*?>")
MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
TEXT_COLUMN = "description"
OPENAI_KEY = "..."
