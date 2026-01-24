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
- **Dynamic Personas**: Users are simulated with dynamically generated personas (e.g., "Lazy/Functional", "Trolling", "Melancholic") to ensure maximum variety, rather than fixed mood slots.

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

## Variability & Randomization

To prevent the model from overfitting to a specific conversation structure, we vary parameters at two levels:

### A. Semantic Variability (LLM-Controlled)
The "Director" (Step 1) generates a unique `Scenario` object for every sample, varying:
*   **User Persona**: Demographics and psychological state (e.g., "bored teenager", "stressed academic").
*   **User Context**: The internal motivation triggering constraints (e.g., "needs homework help", "just trolling").
*   **Opening Style**: Linguistic register, syntax, and orthography errors.
*   **Goethe's Perception**: How Goethe *decodes* (or misinterprets) the user's intent based on his 1830s worldview.
*   **Target Emotion**: The specific affect Goethe takes (e.g., "Amused", "Indigant", "Melancholic").

### B. Structural Variability (Randomly Injected)
The "Actor" (Step 2) receives strict, randomized constraints to force diverse interaction patterns:

| Parameter | Probability | Description |
| :--- | :--- | :--- |
| **User Verbosity** | 40% **Lazy**<br>30% **Ranting**<br>30% **Standard** | Forces the user to be mono-syllabic (1-5 words) or overly verbose. |
| **Goethe Verbosity** | 40% **Epigrammatic**<br>30% **Narrative**<br>30% **Standard** | Goethe replies with a single aphorism vs. a short story vs. normal chat. |
| **Greeting** | 50% **Yes**<br>50% **No** | User says "Hi" vs. starting immediately with the topic. |
| **Addressing** | 80% **Generic**<br>20% **Named** | User treats him as a stranger vs. explicitly calling him "Goethe/Wolfgang". |
| **Anachronisms** | 40% **Present**<br>60% **Absent** | User explicitly mentions modern tech (Smartphone, AI) vs. staying timeless. |
| **Goethe's Curiosity** | 30% **Scientist**<br>70% **Sage** | Goethe asks clarifying questions (Scientific interest) vs. confidently interpreting the unknown (Poetic confidence). |


## Usage

Run the generation script:

```bash
pixi run python -m wolfgang_lm.data.synthetic_finetune
```

The script defaults to generating **3,500 samples** distributed as:
- **80% Small Talk/Functional**: Everyday misunderstandings and casual chats (replacing the previous deep talk/chit-chat split).
- **10% Task Refusal**: Goethe refusing to be a tool.
- **10% Safety**: Goethe handling toxicity with dignity.

## Output
Results are saved to `data_clean/dataset_synthetic_conversational.jsonl`.
The generator handles length constraints internally (truncating or discarding dialogues > 1900 chars).
