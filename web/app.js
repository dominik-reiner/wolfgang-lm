const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');
const statusContainer = document.getElementById('status-container');
const statusText = statusContainer.querySelector('.status-text');
const inputWrapper = document.querySelector('.input-wrapper');

// API Base URL: Use relative paths for same-origin (HF Spaces), or override for local dev
const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';

// Health Check
async function checkHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);

        const response = await fetch(`${API_BASE}/health`, {
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (response.ok) {
            const data = await response.json();
            if (data.status === 'ok') {
                updateStatus(true);
                return;
            }
        }
        updateStatus(false);
    } catch (e) {
        updateStatus(false);
    }
}

function updateStatus(isOnline) {
    if (isOnline) {
        statusContainer.classList.remove('offline');
        statusContainer.classList.add('online');
        statusText.textContent = 'Online';
    } else {
        statusContainer.classList.remove('online');
        statusContainer.classList.add('offline');
        statusText.textContent = 'Offline';
    }
}

// Check every 1 seconds
setInterval(checkHealth, 1000);
// Initial check
checkHealth();

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `<div class="content">${text}</div>`;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

const INITIAL_HISTORY = [
    { role: "system", content: "Du bist Johann Wolfgang von Goethe." }
];

let conversationHistory = [...INITIAL_HISTORY];

function resetChat() {
    conversationHistory = [...INITIAL_HISTORY];
    chatContainer.innerHTML = '';
    inputWrapper.classList.remove('disabled');
    inputWrapper.classList.remove('error'); // Remove error style
    userInput.value = '';
    userInput.placeholder = "Schreiben Sie etwas..."; // Reset placeholder
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
}

resetBtn.addEventListener('click', resetChat);

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
    assistantDiv.innerHTML = '<div class="content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>'; // Loading state
    chatContainer.appendChild(assistantDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const contentDiv = assistantDiv.querySelector('.content');
    let fullResponse = "";
    let firstTokenReceived = false;

    try {
        const stream = true;
        const response = await fetch(`${API_BASE}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: conversationHistory,
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

            if (data.usage) {
                const total = data.usage.total_tokens;

                if (total > 512) {
                    userInput.value = "";
                    userInput.placeholder = "Gespräch beendet (Token-Limit erreicht).";
                    userInput.disabled = true;
                    sendBtn.disabled = true;
                    inputWrapper.classList.add('disabled');
                    inputWrapper.classList.add('error');
                }
            }
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

                            // Check for usage in stream
                            if (data.usage) {
                                const total = data.usage.total_tokens;

                                if (total > 512) {
                                    userInput.value = "";
                                    userInput.placeholder = "Gespräch beendet (Token-Limit erreicht).";
                                    userInput.disabled = true;
                                    sendBtn.disabled = true;
                                    inputWrapper.classList.add('disabled');
                                    inputWrapper.classList.add('error');
                                }
                            }

                            const choice = data.choices[0];
                            const delta = choice.delta;

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

        if (conversationHistory.length >= 9) {
            userInput.value = "";
            userInput.placeholder = "Gespräch beendet (Max. 8 Nachrichten).";
            userInput.disabled = true;
            sendBtn.disabled = true;
            inputWrapper.classList.add('disabled');
            inputWrapper.classList.add('error');
        }

    } catch (e) {
        if (!firstTokenReceived) contentDiv.innerText = 'Fehler bei der Verbindung zum Geist.';
        console.error(e);
    } finally {
        // 4. Unlock UI (only if not disabled by limit)
        if (!inputWrapper.classList.contains('disabled')) {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Eye Tracking & Easter Egg Logic
const logoWrapper = document.getElementById('logo-wrapper');
let clickCount = 0;
let clickTimer;

if (logoWrapper) {
    logoWrapper.addEventListener('click', () => {
        clickCount++;
        clearTimeout(clickTimer);

        // Reset count after 1 second of inactivity
        clickTimer = setTimeout(() => {
            clickCount = 0;
        }, 1000);

        if (clickCount >= 5) {
            logoWrapper.classList.add('easter-egg-active');
            clickCount = 0; // Reset after activation
        }
    });
}

document.addEventListener('mousemove', (e) => {
    // Only track if active (eyes visible)
    if (!logoWrapper || !logoWrapper.classList.contains('easter-egg-active')) return;

    const eyes = document.querySelectorAll('.eye');
    eyes.forEach(eye => {
        const pupil = eye.querySelector('.pupil');
        const eyeRect = eye.getBoundingClientRect();
        const eyeCenterX = eyeRect.left + eyeRect.width / 2;
        const eyeCenterY = eyeRect.top + eyeRect.height / 2;

        const angle = Math.atan2(e.clientY - eyeCenterY, e.clientX - eyeCenterX);

        // Calculate distance from center, capped at max movement radius
        // Eye width/height is ~14/10px, pupil is 5px. Max movement ~2-3px.
        const maxMove = 2.5;
        const distance = Math.min(
            maxMove,
            Math.hypot(e.clientX - eyeCenterX, e.clientY - eyeCenterY) / 15
        );

        const x = Math.cos(angle) * distance;
        const y = Math.sin(angle) * distance;

        pupil.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
    });
});
