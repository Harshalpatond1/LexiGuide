from qa_engine import split_sentences


def split_into_chunks(pages):
    """
    Generic chunking for arbitrary documents.

    Creates small overlapping text chunks while preserving
    page information. Works for both structured paragraphs and line receipts.
    """

    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        sentences = split_sentences(text)


        # Create overlapping chunks
        chunk_size = 3
        overlap = 1

        start = 0

        while start < len(sentences):

            chunk_sentences = sentences[
                start:start + chunk_size
            ]

            if not chunk_sentences:
                break

            chunk_text = " ".join(
                chunk_sentences
            )

            chunks.append({
                "page": page["page"],
                "text": chunk_text
            })

            # Move forward while keeping overlap
            start += chunk_size - overlap

    return chunks