# Synthetic Fine-Tuning Data Generation

To prepare **Wolfgang-LM** for modern chatbot interactions, we use synthetic data generation to bridge the gap between 19th-century literature and 21st-century users.

## Pipeline Architecture

The new generation pipeline (`ScenarioBasedGoetheGenerator`) moves away from simple QA pairs to a robust **3-Stage Process**:

### 1. Scenario Generation (The "Director")
Instead of generating dialogues directly, we first generate a **Scenario** object.
- **Goal**: Create a rich, hidden context for the interaction.
- **Structure**:
  - `user_context`: The user's internal state (e.g., "Stressed student needing a quick summary").
  - `first_message`: The **exact** opening line (forced strictly to be informal/slang/typo-ridden).
  - `goethe_perception`: How Goethe *misinterprets* the situation (e.g., "Student asks for summary" -> "Scholar asks for wisdom").
- **Moods**: Users are simulated with 16 distinct moods (e.g., "Lazy/Functional", "Trolling", "Melancholic"), with a 50% chance of a "Neutral/Functional" state to simulate realism.

### 2. Dialogue Drafting (The "Actor")
The LLM generates a dialogue based strictly on the scenario.
- **Stranger Dynamic**: Goethe and the User do **not** know each other. There is no artificial "Hello friend".
- **Friction as a Feature**: We explicitly optimize for **"Talking Past Each Other"**.
  - *User*: "yo whatsapp"
  - *Goethe*: "The world turns in eternal circles..."
  - This friction creates the immersion of a time-traveling poet connecting to the modern web.
- **Reference Injection**: Real dialogue snippets from `gespraeche.jsonl` are injected into the prompt to prime the model's tone.

### 3. The Critic (The "Judge")
An integrated **LLM-as-a-Judge** evaluates every single dialogue before saving.
- **Checklist**:
  - Does the User sound lazy/modern? (No perfect high-school German).
  - Is Goethe staying in character (No "How can I help you?")?
  - Is there positive friction/misunderstanding?
  - Is it short (max 5 turns)?
- **Filter**: Dialogues failing *any* criteria are discarded (0-tolerance policy), ensuring extremely high dataset quality.

## Usage

Run the generation script:

```bash
pixi run python -m wolfgang_lm.data.synthetic_finetune
```

The script defaults to generating **5,500 samples** distributed as:
- **50% Chit-Chat/Functional**: Everyday misunderstandings.
- **30% Deep Talk**: Philosophical alignment.
- **10% Task Refusal**: Goethe refusing to be a tool.
- **10% Safety**: Goethe handling toxicity with dignity.

## Output
Results are saved to `data_clean/dataset_synthetic_conversational.jsonl`.
Then proceed to splitting/preprocessing:

```bash
pixi run python -m wolfgang_lm.data.split_long_conversations
```
