import os
import glob
import json
import time

try:
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
except ImportError:
    print("Please install the libraries: pip install google-genai python-dotenv")
    exit(1)

# Load environment variables from .env file
load_dotenv()

# Configuration
SOURCE_DIR = "data_clean/gespraeche"
OUTPUT_FILE = "data_clean/gespraeche.jsonl"
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 50000

SYSTEM_PROMPT = """
Du bist ein Experte für Datenextraktion. Deine Aufgabe ist es, Gesprächsverläufe zwischen Johann Peter Eckermann und Johann Wolfgang von Goethe aus dem bereitgestellten Text zu extrahieren.

ZIEL:
Wandle den narrativen Text in **zusammenhängende Dialog-Threads** um. 

REGELN:
1. **Multi-Turn**: Wenn ein Gespräch über mehrere Hin-und-Her-Wechsel geht (Frage -> Antwort -> Nachfrage -> Antwort), fasse dies als EINE Konversation zusammen.
2. **Narrativ zu Dialog**: Wandle Eckermanns Erzählungen ("Ich fragte ihn...") in direkte Rede um ("User").
3. **Goethes Stimme**: Behalte Goethes Antworten ("Assistant") so nahe am Original wie möglich.
4. **Sprache**: Zwingend DEUTSCH.
5. **Format**: Gib ein JSON-Objekt zurück mit dem Key "threads". Jeder Thread ist eine LISTE von Objekten mit "role" ("user" oder "assistant") und "content".

BEISPIEL:
Text:
"Wir sprachen über Schiller. Ich bemerkte, dass er nie ruhte. Goethe stimmte zu und fügte hinzu, dass dies sein Schicksal war. Ich fragte, ob er glücklich war. Goethe verneinte dies."

Output:
{
  "threads": [
    [
      {"role": "user", "content": "Wir sprachen über Schiller. Er war ein Mann, der nie ruhte."},
      {"role": "assistant", "content": "Das ist wahr, das war sein Schicksal."},
      {"role": "user", "content": "War er denn glücklich?"},
      {"role": "assistant", "content": "Nein, das war er nicht."}
    ]
  ]
}
"""


def chunk_text(text, size=CHUNK_SIZE, overlap=1000):
    """
    Smart chunking by accumulating paragraphs.
    Includes overlap to preserve context across potential cuts.
    """
    paragraphs = text.split("\n")

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        if current_length + len(para) > size:
            full_chunk_text = "\n".join(current_chunk)
            chunks.append(full_chunk_text)

            overlap_buffer = []
            overlap_len = 0
            for prev_para in reversed(current_chunk):
                if overlap_len + len(prev_para) > overlap:
                    break
                overlap_buffer.insert(0, prev_para)
                overlap_len += len(prev_para)

            current_chunk = overlap_buffer + [para]
            current_length = overlap_len + len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def extract_from_chunk(client, chunk):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=chunk,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        content = response.text
        if not content:
            return []

        data = json.loads(content)
        return data.get("threads", [])
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return []


def main():
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=API_KEY)

    files = glob.glob(os.path.join(SOURCE_DIR, "*.txt"))
    print(f"Found {len(files)} files to process with {MODEL_NAME}.")

    total_threads = 0
    total_turns = 0

    # Overwrite the file fresh
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for filepath in files:
            print(f"Processing {filepath}...")
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            print(f"  - Split into {len(chunks)} large chunks.")

            for i, chunk in enumerate(chunks):
                print(f"    - Processing Chunk {i+1}/{len(chunks)}...")
                threads = extract_from_chunk(client, chunk)

                if threads:
                    print(f"      > Found {len(threads)} conversation threads.")
                    for thread in threads:
                        # Construct the full chat object
                        # Ensure alternating User/Assistant if possible, but strict extraction is priority

                        # Add System Prompt at the start of every thread
                        full_messages = [
                            {
                                "role": "system",
                                "content": "Du bist Johann Wolfgang von Goethe.",
                            }
                        ]

                        # Add the thread turns
                        full_messages.extend(thread)

                        training_example = {"messages": full_messages}

                        f_out.write(
                            json.dumps(training_example, ensure_ascii=False) + "\n"
                        )
                        f_out.flush()

                        total_threads += 1
                        total_turns += len(thread)

                time.sleep(2)

    print(f"Extraction complete.")
    print(
        f"Saved {total_threads} threads containing {total_turns} individual turns to {OUTPUT_FILE}."
    )


if __name__ == "__main__":
    main()
