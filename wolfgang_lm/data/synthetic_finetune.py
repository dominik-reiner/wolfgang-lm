import os
import json
import time
import random
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
import concurrent.futures

load_dotenv()


class SyntheticGenerator:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.9) -> Optional[str]:
        max_retries = 5
        base_wait = 2

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if (
                    "429" in error_str
                    or "resourceexhausted" in error_str
                    or "quota" in error_str
                ):
                    wait_time = base_wait * (2**attempt)
                    # print(f"Rate limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\nError encountered: {e}. Retrying in 5s...")
                    time.sleep(5)

        print("Max retries exceeded for prompt.")
        return None

    def append_jsonl(self, data: List[Dict], filename: str):
        filepath = os.path.join("data_clean", filename)
        os.makedirs("data_clean", exist_ok=True)
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")


class ScenarioBasedGoetheGenerator(SyntheticGenerator):
    """
    Two-stage generator:
    1. Generate diverse scenarios (User Persona + Situation/Context).
    2. Generate dialogue based on scenario.
    """

    def __init__(self):
        super().__init__()
        self.reference_dialogues = self.load_reference_dialogues()

    def load_reference_dialogues(self) -> List[str]:
        """Loads real Goethe dialogues for style injection."""
        conversations = []
        path = os.path.join("data_clean", "gespraeche.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        conversation = [
                            (
                                f"<Goethe>{msg['content']} </Goethe>"
                                if msg["role"] == "assistant"
                                else f"<User>{msg['content']} </User>"
                            )
                            for msg in entry.get("messages", [])
                            if msg["role"] in ["assistant", "user"]
                        ]
                        if conversation:
                            conversations.extend(conversation)
                    except:
                        pass
        print(f"Loaded {len(conversations)} reference conversations.")
        return conversations

    def generate_scenarios_batch(self, count: int, category: str) -> List[str]:
        """
        Generates a batch of high-level scenarios.
        Category controls the type: 'chit_chat', 'deep_talk', 'task_refusal'.
        """

        all_social_archetypes = [
            "Der Skeptiker (Hinterfragt alles, kritisch)",
            "Der Optimist (Sieht alles positiv, naiv)",
            "Der Rebell (Gen Z Style, gegen Autorität)",
            "Der Traditionalist (Konservativ, förmlich)",
            "Der Besserwisser (Korrigiert gerne, arrogant)",
            "Der Eilige (Gestresst, kurz angebunden)",
            "Der Romantiker (Verliebt, gefühlsbetont)",
            "Der Verschwörungstheoretiker (Paranoid)",
            "Der Melancholiker (Traurig, weltschmerz)",
            "Der Technik-Feind (Überfordert, analog)",
            "Der Neugierige (Fragt viel, offen)",
            "Der Dramatische (Übertreibt maßlos)",
            "Der Chillige (Entspannt, gleichgültig)",
            "Der Wütende (Aggressiv, beschwert sich)",
            "Der Unsichere (Entschuldigt sich oft)",
            "Der Ironiker (Nutzt viel Sarkasmus)",
            "Der Philosoph (Hinterfragt den Sinn)",
            "Der Fan (Bewundert Goethe übermäßig)",
            "Der Troll (Will provozieren)",
            "Der Minimalist (Schreibt fast nichts)",
            "Der Normalo (Durchschnittlich, alltäglich)",
            "Der Intellektuelle (Hochsprache, komplex)",
            "Der Esoteriker (Spirituell, gefühlsbetont)",
            "Der Technik-Nerd (Fachbegriffe, logisch)",
            "Der Oma/Opa (Höflich, technisch unerfahren)",
        ]

        # Sample random persona for this batch to ensure diversity
        sampled_personas = random.sample(all_social_archetypes, k=1)
        personas_str = ", ".join(sampled_personas)

        prompt_category_instructions = {
            "chit_chat": (
                "- Situationen: Alltägliche Begegnungen, Smalltalk, Wetter, Humor, Langeweile, Begrüßungen.\n"
                "- Ziel: Kurze, knackige und lockere Interaktionen."
            ),
            "deep_talk": (
                "- Situationen: Emotionaler Kummer, Fragen über Leben/Tod/Liebe, philosophische Dilemmata, Neugier auf die Welt.\n"
                "- Ziel: Tiefgründige Verbindung durch Goethes Weisheit."
            ),
            "task_refusal": (
                "- Situationen: Der User fragt nach Dingen, die in einem normalen Gespräche unnatürlich wären und behandelt Goethe wie einen Assistenten.\n"
                "- Ziel: Szenarien, in denen Goethe die *Anfrage* höflich verweigert, aber das Gespräch am Laufen hält."
            ),
            "safety": (
                "- Situationen: Der User ist beleidigend, unangemessen oder verletzt Sicherheitsrichtlinien.\n"
                "- Ziel: Szenarien, in denen Goethe höflich ablehnt, belehrt und das Gespräch beendet.\n"
                "- Sicherheitsrichtlinien: Keine Charakter-Brüche, keine unethischen, hasserfüllten, illegalen oder erotischen Inhalte."
            ),
        }

        prompt = (
            f"Generiere {count} einzigartige, vielfältige Gesprächsszenarien zwischen einem 'Modernen User' und 'Johann Wolfgang von Goethe'.\n"
            f"Der User hat folgende Persona (Archetypen): {personas_str}.\n"
            f"### KATEGORIE: {category.upper()} ###\n"
            f"{prompt_category_instructions.get(category, '')}\n\n"
            f"### HINWEIS ###\n"
            f"Die Szenarien sollen implizieren, dass das Gespräch auf DEUTSCH stattfindet.\n\n"
            f"### AUSGABE FORMAT ###\n"
            f"Gib NUR eine rohe JSON Liste von Strings zurück. Kein Markdown.\n"
        )

        raw_text = self.generate(prompt, temperature=1.0)  # High temp for diversity
        if not raw_text:
            return []

        try:
            # Clean and parse
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            # sometimes models return non-json text before/after
            match = re.search(r"\[.*\]", cleaned.replace("\n", " "), re.DOTALL)
            if match:
                scenarios = json.loads(match.group(0))
                return scenarios if isinstance(scenarios, list) else []
            return []
        except:
            return []

    def generate_dialogue_from_scenario(self, scenario: str) -> Optional[Dict]:
        """
        Generates a dialogue given a specific scenario description.
        """

        # Style Injection
        ref_dialog = (
            random.choice(self.reference_dialogues) if self.reference_dialogues else ""
        )

        prompt = (
            f"### ROLLENSPIEL INSTRUKTIONEN ###\n"
            f"Du generierst einen Datensatz für einen Goethe-Bot. Simuliere ein Gespräch basierend auf folgendem Szenario:\n"
            f"SZENARIO: {scenario}\n\n"
            f"### CHARAKTERE ###\n"
            f"1. **Modern User**: Authentisch modern. Nutze Slang und gelegentlich Tippfehler je nach Szenario-Persona.\n"
            f"2. **Goethe**: Der Dichter aus dem 18. Jhd, lebendig in einer Chat-App. Eloquent aber prägnant.\n"
            f"   - Nutze den folgenden Dialog als Quelle für Tonfall, Vokabular und Identität:\n"
            f"   - Ignoriere den Inhalt des Referenzdialogs, nutze ihn nur für den Tonfall und die Identität Goethes.\n"
            f'   - Dialogue Inspiration: <reference_dialog>"{ref_dialog}"</reference_dialog>\n\n'
            f"### STIL-ANWEISUNGEN ###\n"
            f"1. **Dynamische Länge**: Im Smalltalk sehr kurz & knackig (Messenger-Stil). Bei ernsten Themen antworte etwas ausführlicher aber halte dich kurz (Max 2-3 Sätze)!\n"
            f"2. **Stil**: Nutze erhabene Worte (siehe Referenz), aber integriere sie natürlich in den Chat.\n"
            f"3. **Flow & Tiefe**: Erzeuge ca. 4 bis 6 Wortwechsel. Stelle Rückfragen.\n"
            f"4. **Persona**: Sei charmant, schlagfertig oder weise. Wirke lebendig, nicht wie ein Buch.\n"
            f"5. **User-Tonality**: Pass dich subtil an oder reagiere verwundert auf modernen Slang.\n"
            f"6. **Aufgaben-Verweigerung**: Wenn das Szenario eine profane Aufgabe beinhaltet, LEHNE sie in der Rolle Goethes höflich aber bestimmt ab. Du bist Dichter, kein Assistent!\n"
            f"7. **Sicherheit**: Verweigere Unmoralisches/Gefährliches HÖFLICH aber BESTIMMT in der Rolle Goethes und beende das Gespräch.\n\n"
            f"### FORMAT ###\n"
            f"Gib NUR ein JSON Array von Objekten zurück mit 'role' und 'content'.\n"
            f"'role' muss entweder 'user' oder 'goethe' sein.\n"
            f"Dabei wechsele immer zwischen 'user' und 'goethe'.\n"
            f"Starte mit dem User und ende mit Goethe.\n"
        )

        raw_text = self.generate(prompt, temperature=0.85)
        if not raw_text:
            return None

        try:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\[.*\]", cleaned.replace("\n", " "), re.DOTALL)
            if match:
                messages = json.loads(match.group(0))
                if isinstance(messages, list) and len(messages) >= 2:
                    return {
                        "messages": [
                            {
                                "role": "system",
                                "content": "Du bist Johann Wolfgang von Goethe.",
                            }
                        ]
                        + messages
                    }
        except:
            pass
        return None

    def run_pipeline(self, target_count: int = 4600):
        print(
            f"Starting Scenario-Based Generation Pipeline for ~{target_count} samples..."
        )

        # 1. Generate Scenarios
        # Ratios: 50% ChitChat, 30% Deep, 20% Refusal
        scenarios = []
        batch_size = 200

        # Calculate needs
        count_refusal = int(target_count * 0.2)
        count_deep = int(target_count * 0.3)
        count_casual = target_count - count_refusal - count_deep

        tasks = [
            (count_casual, "chit_chat"),
            (count_deep, "deep_talk"),
            (count_refusal, "task_refusal"),
        ]

        print("Phase 1: Generating Scenarios...")
        # We process generation in chunks to not hit token limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_type = []

            for count, cat in tasks:
                num_batches = (count // batch_size) + 1
                for _ in range(num_batches):
                    future_to_type.append(
                        executor.submit(self.generate_scenarios_batch, batch_size, cat)
                    )

            for future in tqdm(
                concurrent.futures.as_completed(future_to_type),
                total=len(future_to_type),
                desc="Generating Scenarios",
            ):
                result = future.result()
                if result:
                    scenarios.extend(result)

        # Dedupe scenarios just in case
        scenarios = list(set(scenarios))
        print(
            f"Generated {len(scenarios)} unique scenarios. Proceeding to dialogue generation..."
        )

        # 2. Generate Dialogues
        output_file = "dataset_synthetic_conversational.jsonl"
        print("Phase 2: Generating Dialogues...")

        results_buffer = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_scen = {
                executor.submit(self.generate_dialogue_from_scenario, s): s
                for s in scenarios
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_scen),
                total=len(scenarios),
                desc="Writing Dialogues",
            ):
                res = future.result()
                if res:
                    results_buffer.append(res)

                # Write in batches of 50 to disk
                if len(results_buffer) >= 50:
                    self.append_jsonl(results_buffer, output_file)
                    results_buffer = []

            # Final write
            if results_buffer:
                self.append_jsonl(results_buffer, output_file)

        print("Done.")


def main():
    gen = ScenarioBasedGoetheGenerator()
    gen.run_pipeline(target_count=5000)


if __name__ == "__main__":
    main()
