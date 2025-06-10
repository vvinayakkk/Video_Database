from memvid import MemvidEncoder, MemvidChat

# Create some sample text chunks
chunks = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "Machine learning is a subset of artificial intelligence that focuses on training models to learn from data.",
    "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
    "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language."
]

# Initialize the encoder
encoder = MemvidEncoder()

# Add chunks to the encoder
encoder.add_chunks(chunks)

# Build the video memory
print("Building video memory...")
encoder.build_video("test_memory.mp4", "test_index.json")
print("Video memory built successfully!")

# Start a chat session
print("\nStarting chat session...")
chat = MemvidChat("test_memory.mp4", "test_index.json")
chat.start_session()

# Test some queries
test_queries = [
    "What is Python?",
    "Tell me about machine learning",
    "What is deep learning?",
    "Explain NLP"
]

print("\nTesting queries:")
for query in test_queries:
    print(f"\nQuery: {query}")
    response = chat.chat(query)
    print(f"Response: {response}")

print("\nTest completed successfully!") 