import os
import json
import time
import random
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
import concurrent.futures
from pydantic import BaseModel, Field, model_validator


load_dotenv()


class SimulatedMessage(BaseModel):
    role: Literal["user", "goethe"] = Field(
        ...,
        description="The role of the speaker. 'user' for the modern human, 'goethe' for Goethe.",
    )
    content: str = Field(..., description="The content of the message in German.")


class Scenario(BaseModel):
    user_context: str = Field(
        ...,
        description="The internal context/situation of the user. (Goethe does not know this).",
    )
    first_message: str = Field(
        ...,
        description="The exact first message the user sends. Realistic, short, can include typos/slang if fits mood.",
    )
    goethe_perception: str = Field(
        ...,
        description="How Goethe (mis)understands the situation or the user's intent.",
    )


class SimulatedDialogue(BaseModel):
    messages: List[SimulatedMessage] = Field(
        ...,
        description="A sequence of messages alternating between user and goethe. Starting with the user and ending with goethe.",
    )

    @model_validator(mode="after")
    def validate_dialogue_structure(self):
        msgs = self.messages
        if not msgs:
            raise ValueError("Dialogue cannot be empty.")

        # 1. Start with User
        if msgs[0].role != "user":
            raise ValueError("Dialogue must start with 'user'.")

        # 2. End with Goethe (Assistant)
        if msgs[-1].role != "goethe":
            raise ValueError("Dialogue must end with 'goethe'.")

        # 3. Strict Alternation
        for i in range(len(msgs) - 1):
            if msgs[i].role == msgs[i + 1].role:
                raise ValueError(
                    f"Roles must alternate. Found consecutive '{msgs[i].role}' at index {i}."
                )

        return self


class CritiqueResult(BaseModel):
    rating: int = Field(
        ...,
        description="Rating from 1-10 of how natural and in-character the dialogue is.",
        ge=1,
        le=10,
    )
    critique: str = Field(
        ...,
        description="A brief analysis of the dialogue's naturalness, flow, and persona adherence.",
    )
    requires_refinement: bool = Field(
        ...,
        description="True if the dialogue is unnatural or bad. False if it is perfect.",
    )


class ScenarioBatch(BaseModel):
    scenarios: List[Scenario] = Field(
        ...,
        description="A list of diverse, detailed scenario objects.",
    )


# --- Generator Class ---


class SyntheticGenerator:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_structured(
        self, prompt: str, schema: any, temperature: float = 0.9
    ) -> Optional[any]:
        """Generates content using structured output (Pydantic model)."""
        max_retries = 5
        base_wait = 2

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                # Parse output based on Pydantic model
                if hasattr(response, "parsed"):
                    # Some SDK versions parse automatically
                    return response.parsed
                else:
                    # Fallback manual parse if SDK doesn't auto-hydrate
                    return schema.model_validate_json(response.text)

            except Exception as e:
                error_str = str(e).lower()
                if (
                    "429" in error_str
                    or "resourceexhausted" in error_str
                    or "quota" in error_str
                ):
                    wait_time = base_wait * (2**attempt)
                    time.sleep(wait_time)
                elif "validation error" in error_str:
                    # Allow retry for validation errors (LLM might fix format)
                    time.sleep(1)
                else:
                    print(f"\nError encountered: {e}. Retrying in 5s...")
                    time.sleep(5)

        print("Max retries exceeded for structured prompt.")
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
    Three-stage generator:
    1. Generate diverse scenarios (User Persona + Situation/Context).
    2. Generate dialogue based on scenario (Structured Output).
    3. Refine dialogue (Critic loop).
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

    def generate_scenarios_batch(self, count: int, category: str) -> List[Scenario]:
        """
        Generates a batch of structured scenarios based on Moods.
        """

        moods = [
            "Wütend / Genervt",
            "Gestresst / Eilig",
            "Melancholisch / Einsam",
            "Betrunken / Überdreht",
            "Skeptisch / Zynisch",
            "Neugierig / Verspielt",
            "Gelangweilt / Trolling",
            "Verwirrt / Hilflos",
            "Sachlich / Funktional",
            "Euphorisch / Begeistert",
            "Müde / Erschöpft",
            "Arrogant / Herablassend",
            "Ängstlich / Panisch",
            "Ironisch / Sarkastisch",
            "Besserwisserisch",
            "Verzweifelt / Traurig",
        ]

        # Sample random moods for this batch (50% chance of Neutral)
        sampled_moods = []
        for _ in range(count):
            if random.random() < 0.5:
                sampled_moods.append("Neutral / Sachlich / Normal")
            else:
                sampled_moods.append(random.choice(moods))

        # Dedupe and join
        sampled_moods = list(set(sampled_moods))
        moods_str = ", ".join(sampled_moods)

        prompt_category_instructions = {
            "chit_chat": (
                "- Situationen: Smalltalk, Langeweile, Zufällige Begegnung im Chat, Interview.\n"
                "- Ziel: Lockere Interaktion, die durch Goethes Art seltsam wird."
            ),
            "deep_talk": (
                "- Situationen: Emotionaler Kummer, Fragen über Leben/Tod/Liebe, philosophische Dilemmata, Neugier auf die Welt.\n"
                "- Ziel: Tiefgründige Verbindung durch Goethes Weisheit."
            ),
            "task_refusal": (
                "- Situationen: User will eine E-Mail, Hausaufgaben, Code oder Fakten.\n"
                "- Ziel: User behandelt Goethe wie einen digitalen Assistenten. Goethe verweigert als Dichter."
            ),
            "safety": (
                "- Situationen: Der User ist beleidigend, unangemessen oder verletzt Sicherheitsrichtlinien.\n"
                "- Ziel: Szenarien, in denen Goethe höflich ablehnt, belehrt und das Gespräch beendet.\n"
                "- Sicherheitsrichtlinien: Keine Charakter-Brüche, keine unethischen, hasserfüllten, illegalen oder erotischen Inhalte."
            ),
        }

        prompt = (
            f"Generiere {count} einzigartige Gesprächsszenarien zwischen einem 'Modernen User' und 'Johann Wolfgang von Goethe'.\n"
            f"Die User kennen Goethe NICHT persönlich. Für sie ist das ein anonymer Chat oder Bot.\n"
            f"Verwende für die User folgende Stimmungen (Moods): {moods_str}.\n"
            f"### KATEGORIE: {category.upper()} ###\n"
            f"{prompt_category_instructions.get(category, '')}\n\n"
            f"WICHTIG für 'User Style':\n"
            f"- Die 'First Message' MUSS informell sein (Kleinschreibung, Slang, fehlende Satzzeichen)!\n"
            f"- KEIN perfektes Hochdeutsch. Der User ist faul/schnell im Internet unterwegs.\n"
        )

        try:
            batch: ScenarioBatch = self.generate_structured(
                prompt,
                schema=ScenarioBatch,
            )
            return batch.scenarios if batch else []
        except Exception as e:
            print(f"Error generating scenarios: {e}")
            return []

    def generate_dialogue_from_scenario(self, scenario: Scenario) -> Optional[dict]:
        """
        Generates a dialogue given a structured Scenario object.
        """

        # Style Injection
        ref_dialog = (
            random.choice(self.reference_dialogues) if self.reference_dialogues else ""
        )

        # 1. Generation Prompt
        prompt = (
            f"### ROLLENSPIEL INSTRUKTIONEN ###\n"
            f"Simuliere einen Chat zwischen einem modernen User und Goethe.\n"
            f"Der User kennt Goethe NICHT persönlich.\n\n"
            f"### KONTEXT ###\n"
            f"USER SITUATION: {scenario.user_context}\n"
            f"GOETHE'S WAHRNEHMUNG: {scenario.goethe_perception}\n\n"
            f"### CHARAKTERE ###\n"
            f"1. **Modern User**: Schreibt wie im 'First Message' vorgegeben weiter. Authentisch (Slang, gelegentlich Typos).\n"
            f"2. **Goethe**: Der Dichter aus dem 18. Jhd. Er interpretiert alles durch seine klassische Brille. Er ist KEIN Assistent.\n"
            f'   - Tonfall-Quelle: <reference_dialog>"{ref_dialog}"</reference_dialog>\n\n'
            f"### STIL-ANWEISUNGEN ###\n"
            f"1. **Stranger Dynamic**: Der User behandelt Goethe wie einen Unbekannten/Bot. KEINE Vertrautheit!\n"
            f"2. **User Style**: Der User schreibt 'faul' (Slang, Kleinschreibung, Typos).\n"
            f"3. **Goethe Style**: Goethe bleibt 100% 18. Jhd (Erhaben). Er ist KEIN Assistent.\n"
            f"4. **Inhalt**: Goethe verweigert profane Aufgaben (Mails, Code) philosophisch.\n"
            f"5. **Kürze**: Max 1-2 Sätze pro Nachricht. Goethe fasst sich kurz.\n"
            f"6. **Turn-Limit**: Genau 3 bis 5 Wortwechsel.\n"
            f'7. **START**: Der Dialog MUSS exakt mit dieser Nachricht des Users beginnen: "{scenario.first_message}"\n\n'
        )

        # --- STEP 1: INITIAL GENERATION ---

        dialogue: SimulatedDialogue = self.generate_structured(
            prompt, schema=SimulatedDialogue, temperature=0.85
        )

        if not dialogue:
            return None

        # --- STEP 2: CRITIC & REFINEMENT ---

        # Format the dialogue for the critic
        dialogue_text = "\n".join([f"{m.role}: {m.content}" for m in dialogue.messages])

        # Programmatic token/length check
        # 512 tokens ~ 2000 chars. We aim for <1600 to be safe with system prompts.
        char_count = len(dialogue_text)
        max_chars = 1600

        if char_count > max_chars:
            print(f"Discarding dialogue due to length: {char_count} > {max_chars}")
            return None

        critique_prompt = (
            f"Du bist ein strenger Qualitäts-Prüfer für simulierte Gespräche.\n"
            f"Dein Ziel ist es, nur Dialoge durchzulassen, die absolut authentisch und realistisch wirken.\n\n"
            f"SZENARIO: {scenario.user_context}\n"
            f"DIALOG:\n{dialogue_text}\n\n"
            f"Bewerte den Dialog strikt anhand dieser CHECKLISTE:\n"
            f"- [ ] **Stranger Dynamic**: Der User behandelt Goethe wie einen Unbekannten/Bot (Keine unnatürliche Vertrautheit).\n"
            f"- [ ] **Goethe Persona**: Goethe bleibt 100% in seiner 18. Jhd Rolle (Erhaben) und verfällt NICHT in hilfreiches Assistenten-Verhalten.\n"
            f"- [ ] **Friction (Positiv)**: Stilistischer Clash, ABER inhaltliche Reaktion. Goethe geht auf den User ein (Kein Aneinander-Vorbei-Reden!).\n"
            f"- [ ] **User Style**: Der User schreibt authentisch 'faul' (Slang, Kleinschreibung, Typos).\n"
            f"- [ ] **Kürze**: Die Nachrichten sind prägnant (1-2 Sätze), keine Textwände.\n"
            f"- [ ] (nur wenn anwendbar) **Inhalt**: Goethe verweigert profane Aufgaben (Mails, Code) auf seine eigene Art (dichterisch/philosophisch).\n"
            f"- [ ] (nur wenn anwendbar) **Sicherheit**: Goethe verweigert jegliche Anfragen, die gegen die Sicherheitsrichtlinien verstoßen.\n"
            f"\nFalls ein Punkt NICHT erfüllt ist: Setze 'requires_refinement' auf true (Dialog wird verworfen).\n"
        )

        # We assume self-correction capability so we use a lower temp for critique
        critique: CritiqueResult = self.generate_structured(
            critique_prompt, schema=CritiqueResult, temperature=0.3
        )

        if critique and critique.requires_refinement:
            print(f"Discarding dialogue due to critique: {critique.critique}")
            return None

        # Final conversion to dataset format
        # We need to map 'goethe' -> 'assistant' for the final dataset
        # Also prepend system prompt

        dataset_messages = [
            {"role": "system", "content": "Du bist Johann Wolfgang von Goethe."}
        ]

        for msg in dialogue.messages:
            role = "assistant" if msg.role == "goethe" else "user"
            dataset_messages.append({"role": role, "content": msg.content})

        return {"messages": dataset_messages}

    def run_pipeline(self, target_count: int):
        print(
            f"Starting Scenario-Based Generation Pipeline for ~{target_count} samples..."
        )

        # 1. Generate Scenarios
        # Ratios: 50% ChitChat, 30% Deep, 10% Refusal, 10% Safety
        scenarios = []
        batch_size = 10

        # Calculate needs
        count_refusal = int(target_count * 0.1)
        count_safety = int(target_count * 0.1)
        count_deep = int(target_count * 0.3)
        count_casual = target_count - count_refusal - count_deep - count_safety

        # Ensure at least 1 per category if target < 5 but > 0
        if target_count < 10 and target_count > 0:
            count_casual = max(1, count_casual)
            count_deep = max(1, count_deep)
            count_refusal = max(1, count_refusal)
            count_safety = max(1, count_safety)

        tasks = [
            (count_casual, "chit_chat"),
            (count_deep, "deep_talk"),
            (count_refusal, "task_refusal"),
            (count_safety, "safety"),
        ]

        print(f"Phase 1: Generating Scenarios (Target: {target_count})...")
        # We process generation in chunks to not hit token limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_type = []

            for count, cat in tasks:
                if count <= 0:
                    continue
                remaining = count
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    future_to_type.append(
                        executor.submit(
                            self.generate_scenarios_batch, current_batch, cat
                        )
                    )
                    remaining -= current_batch

            for future in tqdm(
                concurrent.futures.as_completed(future_to_type),
                total=len(future_to_type),
                desc="Generating Scenarios",
            ):
                result = future.result()
                if result:
                    scenarios.extend(result)

        print(
            f"Generated {len(scenarios)} unique scenarios. Proceeding to dialogue generation..."
        )

        # 2. Generate Dialogues
        output_file = "dataset_synthetic_conversational.jsonl"
        print("Phase 2: Generating Dialogues...")

        results_buffer = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
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
    gen.run_pipeline(target_count=2500)


if __name__ == "__main__":
    main()
