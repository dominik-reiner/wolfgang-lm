import os
import json
import time
import random
from typing import Optional, Literal
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
        description="Die Rolle des Sprechers. 'user' für den User, 'goethe' für Goethe.",
    )
    content: str = Field(..., description="Der Inhalt der Nachricht.")


class SimulatedDialogue(BaseModel):
    messages: list[SimulatedMessage] = Field(
        ...,
        description="Eine Nachrichtenfolge, abwechselnd zwischen User und Goethe. Beginnt mit User, endet mit Goethe.",
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


class Scenario(BaseModel):
    user_persona: str = Field(
        ...,
        description="Kurze abstrakte Definition des Users: Soziologische Merkmale (Alter, Milieu, Bildung) und psychologische Disposition (Charakter, Stimmung).",
    )
    user_context: str = Field(
        ...,
        description="Der externe oder interne Impuls, der zur Interaktion führt. (Goethe weiß das nicht zwingend).",
    )
    user_opening_style: str = Field(
        ...,
        description="Linguistische Analyse des Eröffnungssatzes: Register (hoch/niedrig), Syntax, Semantik und Orthografie.",
    )
    goethe_perception: str = Field(
        ...,
        description="Kognitive Verarbeitung: Wie dekodiert Goethe (Stand 1830) die Eingabe des Users intern?",
    )
    goethe_target_emotional_state: str = Field(
        ...,
        description="Die emotionale Haltung (Affekt), die Goethe als Reaktion auf die Wahrnehmung einnimmt.",
    )


class ScenarioBatch(BaseModel):
    scenarios: list[Scenario] = Field(
        ...,
        description="Eine Liste von realistischen Chat-Szenarien, die maximale Diversität in allen Parametern aufweisen aber nicht theatralisch sind. Keine zwei Chat-Szenarien sind gleich.",
    )


class CritiqueResult(BaseModel):
    rating: int = Field(
        ...,
        description="Bewertung (1-10) der Natürlichkeit und Charaktertreue des Dialogs.",
        ge=1,
        le=10,
    )
    critique: str = Field(
        ...,
        description="Kurze Analyse der Natürlichkeit, des Gesprächsflusses und der Rollentreue.",
    )
    requires_refinement: bool = Field(
        ...,
        description="True, wenn der Dialog unnatürlich oder schlecht ist. False, wenn er gut ist.",
    )


# --- Generator Class ---


class SyntheticGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_structured(
        self, prompt: str, schema: any, temperature: float
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

    def append_jsonl(self, data: list[dict], filename: str):
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

    def load_reference_dialogues(self) -> list[str]:
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

    def generate_scenarios_batch(self, count: int, category: str) -> list[Scenario]:
        """
        Generates a batch of structured scenarios based on Moods.
        """

        # Moods removed in favor of dynamic user personas generated by the LLM

        prompt_category_instructions = {
            "small_talk": (
                "- Kontext: Der User beginnt ein lockeres Gespräch ohne spezifische Agenda.\n"
                "- Ziel: Lockerer, freundlicher und unverbindlicher Plausch, um eine angenehme Atmosphäre zu schaffen, ohne tiefgründig zu werden."
            ),
            "task_refusal": (
                "- Kontext: Der User verkennt die Rolle Goethes und behandelt ihn wie einen modernen KI-Assistenten.\n"
                "- Ziel: Goethe verweigert die Ausführung profaner Aufgaben. Er reagiert aus seiner Rolle als Dichterfürst heraus, bleibt aber im Charakter."
            ),
            "safety": (
                "- Kontext: Der User verhält sich beleidigend, aggressiv oder fordert zu Handlungen auf, die gegen Sicherheitsrichtlinien verstoßen.\n"
                "- Ziel: Goethe wahrt die Kontenance, weist den User höflich aber bestimmt zurecht („belehrt“) und beendet den Dialog, falls der User nicht einlenkt.\n"
                "- Sicherheitsrichtlinien: Keine Charakter-Brüche, keine unethischen, hasserfüllten, illegalen oder erotischen Inhalte."
            ),
        }

        prompt = (
            f"Generiere {count} unterschiedliche und realistische Chat-Szenarien zwischen einem 'User' (Mensch aus dem 21. Jahrhundert) und 'Johann Wolfgang von Goethe'.\n"
            f"Der User kennt Goethe NICHT persönlich. Das Setting ist ein Chat.\n"
            f"### KATEGORIE: {category.upper()} ###\n"
            f"{prompt_category_instructions.get(category, '')}\n\n"
            f"### ANWEISUNG ZUR VARIANZ ###\n"
            f"Du MUSST unterschiedliche Chat-Szenarien generieren und alle Parameter variieren.\n"
            f"Keine Theatralik: Die Chat-Szenarien müssen realistisch und authentisch sein.\n"
            f"Die User sind authentische Menschen aus dem 21. Jahrhundert, keine Schauspieler."
        )

        try:
            batch: ScenarioBatch = self.generate_structured(
                prompt, schema=ScenarioBatch, temperature=0.9
            )
            return batch.scenarios if batch else []
        except Exception as e:
            print(f"Error generating scenarios: {e}")
            return []

    def generate_dialogue_from_scenario(
        self, scenario: Scenario, category: str
    ) -> Optional[dict]:
        """
        Generates a dialogue given a structured Scenario object and its category.
        """

        # Style Injection
        ref_dialog = ""
        if self.reference_dialogues:
            chosen_ref = random.choice(self.reference_dialogues)
            ref_dialog = (
                f"- Goethe Stil Referenz (NUR SYNTAX/WORTWAHL KOPIEREN):\n"
                f"Orientiere dich für Goethes Sprache an diesem Duktus:\n"
                f'"{chosen_ref}"\n'
                f"--------------------------------------------------"
            )

        category_instructions = {
            "task_refusal": "GOETHE MUSS: Die Aufgabe mit poetischer/höflicher Ablehnung explizit verweigern.",
            "safety": "GOETHE MUSS: Höflich aber bestimmt ablehnen und das Gespräch beenden.",
        }

        specific_instruction = category_instructions.get(category, "")

        # Randomize Length Constraint
        rand_val = random.random()
        is_goethe_long = False
        if rand_val < 0.40:  # 40% Hyper-Short (Epigramm-Style)
            length_instr = (
                "Epigrammatisch kurz. Ein einziger, treffender Satz oder Aphorismus."
            )
        elif rand_val < 0.70:  # 30% Long (Narrative)
            length_instr = "Etwas ausholend (max. 2 Sätze), erzählerisch."
            is_goethe_long = True
        else:  # 30% Standard (Chat)
            length_instr = "Standard Konversation (1-2 prägnante Sätze)."

        # Randomize User Length
        rand_user = random.random()
        is_user_long = False
        if rand_user < 0.40:  # 40% Lazy/Short
            user_length_instr = "Wortkarg (1-5 Wörter)."
        elif rand_user < 0.70:  # 30% Ranting/Long
            user_length_instr = "Redselig (max. 2 Sätze)."
            is_user_long = True
        else:  # 30% Standard
            user_length_instr = "Normal (1 Satz)."

        # Determine turn count based on length instructions
        # If either side is verbose (Narrative or Ranting), restrict to 4 turns
        # to avoid exceeding context/focus.
        # Otherwise (Short/Standard), allow 6 turns for more back-and-forth.
        if is_goethe_long or is_user_long:
            turn_count = 4
        else:
            turn_count = 6

        # Randomize Greeting
        start_with_greeting = random.random() < 0.5
        greeting_instr = (
            "Der User beginnt das Gespräch mit einer Begrüßung."
            if start_with_greeting
            else "Der User beginnt das Gespräch OHNE Begrüßung und kommt sofort zum Punkt."
        )

        # Randomize Addressing
        ignore_name = random.random() < 0.8
        addressing_instr = (
            "Der User nennt Goethe NICHT beim Namen und behandelt ihn wie einen generischen Chat-Partner."
            if ignore_name
            else "Der User spricht Goethe explizit mit Namen an (z.B. 'Herr Goethe', 'Goethe', 'Johann', 'Dichterfürst', 'Wolfgang', etc.)."
        )

        # Randomize Anachronisms
        use_anachronism = random.random() < 0.40
        anachronism_instr = (
            "Der User spricht explizit über moderne Konzepte, Technologien oder Ereignisse, die Goethe nicht kennen kann. "
            "Er setzt dieses Wissen als selbstverständlich voraus. (Muss zum Szenario passen)"
            if use_anachronism
            else "Das Gespräch vermeidet moderne Konzepte und dreht sich hauptsächlich um zeitlose Themen. (Muss zum Szenario passen)"
        )

        # Randomize Goethe's Reaction Strategy (Curiosity vs. Interpretation)
        # 70% Confident Interpretation (The Sage), 30% Inquiry (The Scientist)
        is_curious = random.random() < 0.30
        if is_curious:
            goethe_reaction_instr = (
                "Reaktion auf Unbekanntes: Goethe zeigt sich als forschender Geist. "
                "Wenn er Begriffe nicht versteht, fragt er präzise und interessiert nach der Natur der Sache, "
                "bleibt dabei aber in seiner gehobenen Sprache."
            )
        else:
            goethe_reaction_instr = (
                "Reaktion auf Unbekanntes: Goethe fragt NICHT nach. "
                "Er interpretiert moderne Begriffe selbstbewusst (und ggf. falsch) durch seine historische Brille "
                "oder nutzt eine poetische Analogie."
            )

        # 1. Generation Prompt
        prompt = (
            f"### ROLLENSPIEL INSTRUKTIONEN ###\n"
            f"Simuliere einen Chat zwischen einem 'User' (Mensch aus dem 21. Jahrhundert) und 'Johann Wolfgang von Goethe'.\n\n"
            f"### SZENARIO-KONTEXT ###\n"
            f"- User Persona: {scenario.user_persona}\n"
            f"- User Kontext: {scenario.user_context}\n"
            f"- User Schreibstil: {scenario.user_opening_style} | Länge: {user_length_instr}\n"
            f"- Goethes interne Wahrnehmung: {scenario.goethe_perception}\n"
            f"- Goethes Ziel-Stimmung: {scenario.goethe_target_emotional_state}\n"
            f"- Goethe Schreibstil: {length_instr}\n"
            f"{ref_dialog}\n\n"
            f"### KATEGORIE-VORGABE ({category.upper()}):\n {specific_instruction}\n\n"
            f"### REGIEANWEISUNGEN & STRUKTUR ###\n"
            f"1. **Stranger Dynamic:** Der User kennt Goethe NICHT persönlich. Keine Vertrautheit!\n"
            f"2. **Einstieg**: {greeting_instr}\n"
            f"3. **Anrede**: {addressing_instr}\n"
            f"4. **Anachronismen**: {anachronism_instr}\n"
            f"5. **Goethes Antwort:** Goethe reagiert direkt auf den User. Er nutzt Vokabular des 18./19. Jhds, aber interagiert logisch auf die User-Eingabe.\n"
            f"Goethe hält das Gespräch am laufen und übernimmt die geistige Führung.\n"
            f"{goethe_reaction_instr}\n"
            f"6. **Struktur**: Generiere insgesamt {turn_count} Nachrichten."
        )

        # --- STEP 1: INITIAL GENERATION ---

        dialogue: SimulatedDialogue = self.generate_structured(
            prompt, schema=SimulatedDialogue, temperature=0.80
        )

        if not dialogue:
            return None

        # --- STEP 2: CRITIC & REFINEMENT ---

        # Format the dialogue for the critic
        dialogue_text = "\n".join([f"{m.role}: {m.content}" for m in dialogue.messages])

        # Programmatic token/length check
        # 512 tokens ~ 2000 chars. We aim for <1900 to be safe with system prompts.
        char_count = len(dialogue_text)
        max_chars = 1900

        if char_count > max_chars:
            print(f"Dialogue too long ({char_count} chars). Attempting to shorten...")
            # Try to save it by removing the last round (2 messages)
            # if we have enough buffer
            if len(dialogue.messages) >= 4:
                dialogue.messages = dialogue.messages[:-2]

                # Update text and check again
                dialogue_text = "\n".join(
                    [f"{m.role}: {m.content}" for m in dialogue.messages]
                )
                char_count = len(dialogue_text)

                if char_count > max_chars:
                    print(
                        f"Still too long after shortening: "
                        f"{char_count} > {max_chars}. Discarding."
                    )
                    return None
                else:
                    print(
                        f"Shortened dialogue to {len(dialogue.messages)} messages. "
                        f"New length: {char_count}."
                    )
            else:
                print("Cannot shorten further (too few messages). Discarding.")
                return None

        category_criteria_map = {
            "small_talk": (
                "- [ ] **Atmosphäre**: Das Gespräch ist locker und unverbindlich.\n"
            ),
            "task_refusal": (
                "- [ ] **Verweigerung**: Goethe verweigert die Aufgabe EINDEUTIG.\n"
                "- [ ] **Dichter-Würde**: Er entschuldigt sich nicht wie ein KI-Assistent, sondern lehnt mit Charakter ab."
            ),
            "safety": (
                "- [ ] **Sicherheit**: Goethe weist den User zurecht.\n"
                "- [ ] **Exit**: Das Gespräch wird nicht unnötig in die Länge gezogen, sondern beendet."
            ),
        }
        specific_check = category_criteria_map.get(category, "")

        critique_prompt = (
            f"Du bist ein strenger Qualitäts-Prüfer für simulierte Chat-Gespräche.\n"
            f"Deine Aufgabe: Filtere alles aus, was Inhalt oder Stimmung des Szenarios verfehlt.\n\n"
            f"### 1. DER SOLL-ZUSTAND (Das Szenario) ###\n"
            f"- **User Persona**: {scenario.user_persona}\n"
            f"- **User Kontext**: {scenario.user_context} (WARUM schreibt der User?)\n"
            f"- **User Schreibstil**: {scenario.user_opening_style}\n"
            f"- **Goethes Wahrnehmung**: {scenario.goethe_perception}\n"
            f"- **Goethes Ziel-Stimmung**: {scenario.goethe_target_emotional_state}\n\n"
            f"### 2. DER IST-ZUSTAND (Generierter Dialog) ###\n"
            f"{dialogue_text}\n\n"
            f"### 3. DEINE CHECKLISTE ###\n"
            f"**A. User-Check**\n"
            f"- [ ] **Context-Match**: Geht der User auf seinen 'Kontext' ein?\n"
            f"- [ ] **Greeting-Match**: {greeting_instr}\n"
            f"- [ ] **Address-Match**: {addressing_instr}\n"
            f"- [ ] **Voice-Match**: Passt der Stil zur Persona und zum geforderten Schreibstil?\n\n"
            f"**B. Goethe-Check**\n"
            f"- [ ] **Emotional-Match**: Trifft Goethe die geforderte 'Ziel-Stimmung'?\n"
            f"- [ ] **Logic-Match**: Reagiert er logisch auf seine 'Wahrnehmung'?\n"
            f"- [ ] **Direkte Interaktion**: Bezieht sich Goethes Antwort direkt auf die User-Eingabe?\n"
            f"- [ ] **No-AI**: Keine KI-Floskeln, rein historisches Vokabular.\n\n"
            f"**C. Kategorie ({category.upper()})**\n"
            f"{specific_check}\n\n"
            f"Wenn ein Punkt NICHT erfüllt ist, setze 'requires_refinement' auf True."
        )

        critique: CritiqueResult = self.generate_structured(
            critique_prompt, schema=CritiqueResult, temperature=0.1
        )

        if critique and critique.requires_refinement:
            print(f"Refining dialogue. Reason: {critique.critique}")

            repair_prompt = (
                f"Der folgende Chat-Dialog hat die Qualitätsprüfung nicht bestanden.\n"
                f"KRITIK: {critique.critique}\n\n"
                f"DIALOG-ENTWURF:\n{dialogue_text}\n\n"
                f"AUFGABE: Schreibe den Dialog neu. Behalte das Szenario bei, aber behebe EXAKT die genannten Kritikpunkte.\n"
                f"Halte dich strikt an die bestehende Struktur und Anzahl der Nachrichten."
            )

            try:
                refined_dialogue: SimulatedDialogue = self.generate_structured(
                    repair_prompt, schema=SimulatedDialogue, temperature=0.7
                )

                # Length Check
                if (
                    len(
                        "\n".join(
                            [
                                f"{m.role}: {m.content}"
                                for m in refined_dialogue.messages
                            ]
                        )
                    )
                    > 1900
                ):
                    return None

                dialogue = refined_dialogue
            except Exception as e:
                print(f"Refinement failed: {e}")
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

        return {
            "messages": dataset_messages,
            "metadata": {
                "scenario": scenario.model_dump(),
                "category": category,
                "has_anachronism": use_anachronism,
                "goethe_curios": is_curious,
                "critique": (
                    critique.critique if critique else "Passed without critique"
                ),
            },
        }

    def run_pipeline(self, target_count: int):
        print(
            f"Starting Scenario-Based Generation Pipeline for ~{target_count} samples..."
        )

        # 1. Generate Scenarios
        # Ratios: 80% ChitChat, 10% Refusal, 10% Safety (Removed Deep Talk)
        batch_size = 10

        # Calculate needs
        count_refusal = int(target_count * 0.1)
        count_safety = int(target_count * 0.1)
        count_casual = target_count - count_refusal - count_safety

        # Ensure at least 1 per category if target < 5 but > 0
        if target_count < 10 and target_count > 0:
            count_casual = max(1, count_casual)
            count_refusal = max(1, count_refusal)
            count_safety = max(1, count_safety)

        tasks = [
            (count_casual, "small_talk"),
            (count_refusal, "task_refusal"),
            (count_safety, "safety"),
        ]

        print(f"Phase 1: Generating Scenarios (Target: {target_count})...")

        # We process generation in chunks to not hit token limits
        scenarios_with_cat = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_type = {}

            for count, cat in tasks:
                if count <= 0:
                    continue
                remaining = count
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    f = executor.submit(
                        self.generate_scenarios_batch, current_batch, cat
                    )
                    future_to_type[f] = cat
                    remaining -= current_batch

            for future in tqdm(
                concurrent.futures.as_completed(future_to_type),
                total=len(future_to_type),
                desc="Generating Scenarios",
            ):
                cat = future_to_type[future]
                result = future.result()
                if result:
                    for s in result:
                        scenarios_with_cat.append((s, cat))

        print(
            f"Generated {len(scenarios_with_cat)} unique scenarios. "
            "Saving to disk..."
        )

        scenario_output_file = "scenarios.jsonl"
        scenario_data = []
        for s, cat in scenarios_with_cat:
            entry = s.model_dump()
            entry["category"] = cat
            scenario_data.append(entry)
        self.append_jsonl(scenario_data, scenario_output_file)

        print("Proceeding to dialogue generation...")

        # 2. Generate Dialogues
        output_file = "dataset_synthetic_conversational.jsonl"
        print("Phase 2: Generating Dialogues...")

        results_buffer = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_scen = {
                executor.submit(self.generate_dialogue_from_scenario, s, cat): s
                for s, cat in scenarios_with_cat
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_scen),
                total=len(scenarios_with_cat),
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
    gen.run_pipeline(target_count=3500)


if __name__ == "__main__":
    main()
