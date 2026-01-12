const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `<div class="content">${text}</div>`;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

let conversationHistory = [
    { role: "system", content: "Du bist Johann Wolfgang von Goethe." }
];

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';

    // Add user message to history
    conversationHistory.push({ role: "user", content: text });

    try {
        const response = await fetch('http://localhost:8000/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: conversationHistory,
                max_tokens: 150
            })
        });

        const data = await response.json();
        const aiText = data.choices[0].message.content;

        appendMessage('assistant', aiText);

        // Add assistant response to history so context grows
        conversationHistory.push({ role: "assistant", content: aiText });

    } catch (e) {
        appendMessage('assistant', 'Fehler bei der Verbindung zum Geist.');
        console.error(e);
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
