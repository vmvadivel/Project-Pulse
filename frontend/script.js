document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');

    // Function to add a message to the chat window
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (sender === 'user') {
            messageDiv.classList.add('user-message');
        } else {
            messageDiv.classList.add('bot-message');
        }
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the bottom
    }

    // Function to handle sending a message
    async function sendMessage() {
        const messageText = userInput.value.trim();

        if (messageText !== "") {
            // Display the user's message
            addMessage(messageText, 'user');
            userInput.value = "";
    
            // Send the message to the backend
            try {
                const response = await fetch('http://127.0.0.1:8000/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: messageText }),
                });
    
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
    
                const data = await response.json();
                
                // Display the bot's response
                addMessage(data.response, 'bot');
    
            } catch (error) {
                console.error('Error:', error);
                addMessage("Sorry, I'm having trouble connecting to the service. Please try again later.", 'bot');
            }
        }
    }

    // Add event listener to the send button
    sendButton.addEventListener('click', sendMessage);

    // Add event listener to handle 'Enter' key press
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial messages for a clean start
    addMessage("Hello! I am your Virtual Buddy.", 'bot');
});