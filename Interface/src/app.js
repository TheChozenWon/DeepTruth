const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const fileInput = document.getElementById('file-input');

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

attachButton.addEventListener('click', function () {
    fileInput.click();
});

fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
        const fileLink = document.createElement('a');
        fileLink.href = URL.createObjectURL(file);
        fileLink.textContent = `Attached: ${file.name}`;
        chatMessages.appendChild(fileLink);
        chatMessages.appendChild(document.createElement('br'));
    }
});

function sendMessage() {
    const messageText = userInput.value.trim();
    if (messageText) {
        const messageElement = document.createElement('div');
        messageElement.textContent = messageText;
        chatMessages.appendChild(messageElement);
        userInput.value = '';
    }
}

function initChat() {
    chatMessages.innerHTML = '';
}

document.addEventListener('DOMContentLoaded', initChat);