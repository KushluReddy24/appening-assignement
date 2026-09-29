import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def main() -> None:
    for query in SAMPLE_QUERIES:
        request = Request(
            "http://127.0.0.1:8000/chat",
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as error:
            raise SystemExit(f"Request failed for {query!r}: {error}") from error

        expected_fields = {
            "query",
            "final_answer",
            "retrieved_context_chunks",
            "confidence_score",
        }
        if set(result) != expected_fields:
            raise SystemExit(f"Unexpected response fields for {query!r}: {result}")
        if query == SAMPLE_QUERIES[-1] and result["confidence_score"] != 0:
            raise SystemExit("Out-of-scope query should be refused with confidence 0.")
        print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()