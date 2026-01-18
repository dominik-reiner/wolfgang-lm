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

    // 1. Lock UI
    userInput.disabled = true;
    sendBtn.disabled = true;

    // 2. Add User Message
    appendMessage('user', text);
    userInput.value = '';  // Clear input
    conversationHistory.push({ role: "user", content: text });

    // 3. Prepare Assistant Message Placeholder
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'message assistant';
    assistantDiv.innerHTML = '<div class="content">...</div>'; // Loading state
    chatContainer.appendChild(assistantDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const contentDiv = assistantDiv.querySelector('.content');
    let fullResponse = "";
    let firstTokenReceived = false;

    try {
        const stream = false;  // Toggle this to enable/disable streaming
        const response = await fetch('http://localhost:8000/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: conversationHistory,
                max_tokens: 150,
                stream: stream
            })
        });

        if (!stream) {
            // Handle Non-Streaming Response
            const data = await response.json();
            const content = data.choices[0].message.content;

            contentDiv.innerHTML = ""; // Clear loading "..."
            contentDiv.innerText = content;
            fullResponse = content;
        } else {
            // Handle Streaming Response (SSE)
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        // Remove "data: " prefix from SSE line
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') break;

                        try {
                            const data = JSON.parse(dataStr);
                            const delta = data.choices[0].delta;

                            if (delta.content) {
                                if (!firstTokenReceived) {
                                    contentDiv.innerHTML = ""; // Clear loading "..."
                                    firstTokenReceived = true;
                                }
                                fullResponse += delta.content;
                                contentDiv.innerText = fullResponse; // Use innerText for safe rendering
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (e) {
                            console.error("Error parsing SSE JSON", e);
                        }
                    }
                }
            }
        }

        // Add assistant response to history so context grows
        conversationHistory.push({ role: "assistant", content: fullResponse });

    } catch (e) {
        if (!firstTokenReceived) contentDiv.innerText = 'Fehler bei der Verbindung zum Geist.';
        console.error(e);
    } finally {
        // 4. Unlock UI
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
