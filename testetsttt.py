import torch
import torchhd.embeddings as embeddings

NUM_EMBEDDINGS = 10  # Small number of hypervectors
DIMENSIONS = 6       # Small dimension size for testing
TEST_INDEX = 3       # Index to test retrieval

random_hvs = embeddings.Random(NUM_EMBEDDINGS, DIMENSIONS)
print(random_hvs)
hv_call = random_hvs(torch.tensor(TEST_INDEX, dtype=torch.long))

hv_direct = random_hvs.weight[TEST_INDEX]

print("Access via function call:\n", hv_call)
print("Access via weight indexing:\n", hv_direct)

# Check if both results are identical
is_identical = torch.allclose(hv_call, hv_direct)
print("\nAre the results identical?:", is_identical)
