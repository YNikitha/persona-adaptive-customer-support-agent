# RAG Configuration

# Number of relevant document chunks to retrieve
TOP_K_RESULTS = 3


# Minimum confidence score required to answer automatically
# If the score is lower, the conversation will be escalated
CONFIDENCE_THRESHOLD = 0.45


# Maximum number of frustrated interactions before escalation
MAX_FRUSTRATION_COUNT = 2


# Sensitive topics that always require human escalation
SENSITIVE_TOPICS = [
    "billing",
    "refund",
    "payment",
    "legal",
    "account deletion",
    "account access"
]


# ChromaDB storage location
CHROMA_DB_PATH = "chroma_db"


# Chunking configuration for RAG
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40