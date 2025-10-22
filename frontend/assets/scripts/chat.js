class ChatInterface {
    constructor() {
        this.apiBase = 'http://127.0.0.1:8000';
        //this.apiBase = '/api';
        this.settings = {
            theme: 'light',
            fontSize: 16,
            autoScroll: true
        };
        
        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        try {
            this.initializeElements();
            this.loadSettings();
            this.setupEventListeners();
            this.setWelcomeTime();
            this.setupConnectionMonitoring();
        } catch (error) {
            console.error('Initialization failed:', error);
        }
    }

    initializeElements() {
        // core elements
        this.userInput = document.getElementById('user-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.charCount = document.getElementById('char-count');
        
        // theme
        this.themeToggle = document.getElementById('theme-toggle');
        this.themeIcon = document.getElementById('theme-icon');
        
        // settings
        this.settingsBtn = document.getElementById('settings-btn');
        this.settingsPanel = document.getElementById('settings-panel');
        this.settingsOverlay = document.getElementById('settings-overlay');
        this.closeSettings = document.getElementById('close-settings');
        this.clearChatBtn = document.getElementById('clear-chat');
        
        // settings controls
        this.fontSizeSelect = document.getElementById('font-size-select');
        this.autoScrollToggle = document.getElementById('auto-scroll');
        this.clearAllDataBtn = document.getElementById('clear-all-data');

        // check if required elements exist
        if (!this.userInput || !this.sendButton || !this.chatMessages) {
            throw new Error('Required DOM elements not found');
        }
    }

    setupEventListeners() {
        // send message
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // input handling
        this.userInput.addEventListener('input', () => {
            this.updateCharCount();
            this.adjustTextareaHeight();
        });

        // theme toggle
        this.themeToggle?.addEventListener('click', () => this.toggleTheme());

        // settings panel
        this.settingsBtn?.addEventListener('click', () => this.showSettings());
        this.closeSettings?.addEventListener('click', () => this.hideSettings());
        this.settingsOverlay?.addEventListener('click', () => this.hideSettings());
        
        this.fontSizeSelect?.addEventListener('change', () => this.updateFontSize());
        this.autoScrollToggle?.addEventListener('change', () => this.updateAutoScroll());
        
        // clear actions
        this.clearChatBtn?.addEventListener('click', () => this.clearChat());
        this.clearAllDataBtn?.addEventListener('click', () => this.clearAllData());

        // prevent settings panel click from closing it
        this.settingsPanel?.addEventListener('click', (e) => e.stopPropagation());
    }

    async sendMessage() {
        const messageText = this.userInput.value.trim();

        if (!messageText) return;

        this.setSendButtonState(false);
        this.addMessage(messageText, 'user');
        
        this.userInput.value = '';
        this.updateCharCount();
        this.adjustTextareaHeight();

        this.showTypingIndicator();

        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: messageText })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            this.hideTypingIndicator();
            this.addMessage(data.response, 'bot', {
                sourceFiles: data.source_files,
                numSources: data.num_sources
            });

        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            
            const errorMessage = error.message.includes('Failed to fetch') 
                ? "I'm having trouble connecting to the server. Please check your connection and try again."
                : `Sorry, I encountered an error: ${error.message}`;
                
            this.addMessage(errorMessage, 'bot', { isError: true });
        } finally {
            this.setSendButtonState(true);
        }
    }

    addMessage(text, sender, options = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const timestamp = this.formatTime(new Date());
        
        const sourceInfo = options.sourceFiles?.length > 0 
            ? `<div class="source-info">📚 Sources: ${options.sourceFiles.join(', ')}</div>` 
            : '';

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <img src="assets/icons/${sender === 'user' ? 'user' : 'bot'}.svg" alt="${sender}" class="avatar-icon">
            </div>
            <div class="message-content">
                <div class="message-bubble ${options.isError ? 'error' : ''}">
                    <div class="message-text">${this.escapeHtml(text)}</div>
                    ${sourceInfo}
                    <div class="message-time">${timestamp}</div>
                </div>
                ${sender === 'bot' ? `
                    <div class="message-actions">
                        <button class="action-btn" onclick="chatInterface.copyMessage(this)">
                            <img src="assets/icons/copy.svg" alt="Copy" width="12" height="12"> Copy
                        </button>
                    </div>
                ` : ''}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        
        if (this.settings.autoScroll) {
            this.scrollToBottom();
        }

        this.applyFontSize();
    }

    copyMessage(button) {
        const messageText = button.closest('.message-content')
            .querySelector('.message-text').textContent;
        
        navigator.clipboard.writeText(messageText)
            .then(() => {
                const originalHTML = button.innerHTML;
                button.innerHTML = '<img src="assets/icons/check.svg" alt="Copied" width="12" height="12"> Copied!';
                button.style.background = 'var(--success-color)';
                button.style.color = 'white';
                
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                    button.style.background = '';
                    button.style.color = '';
                }, 2000);
            })
            .catch(() => {
                button.textContent = 'Unable to copy';
                setTimeout(() => {
                    button.innerHTML = '<img src="assets/icons/copy.svg" alt="Copy" width="12" height="12"> Copy';
                }, 2000);
            });
    }

    showTypingIndicator() {
        this.typingIndicator?.classList.add('show');
        if (this.settings.autoScroll) {
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        this.typingIndicator?.classList.remove('show');
    }

    // theme stuff
    toggleTheme() {
        this.settings.theme = this.settings.theme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        this.saveSettings();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.settings.theme);
        
        // update icon
        if (this.themeIcon) {
            this.themeIcon.src = this.settings.theme === 'light' 
                ? 'assets/icons/sun.svg' 
                : 'assets/icons/moon.svg';
        }
    }

    // settings panel
    showSettings() {
        this.settingsPanel?.classList.add('show');
        this.settingsOverlay?.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    hideSettings() {
        this.settingsPanel?.classList.remove('show');
        this.settingsOverlay?.classList.remove('show');
        document.body.style.overflow = '';
    }

    updateFontSize() {
        this.settings.fontSize = parseInt(this.fontSizeSelect.value);
        this.applyFontSize();
        this.saveSettings();
    }

    applyFontSize() {
        document.querySelectorAll('.message-text').forEach(el => {
            el.style.fontSize = `${this.settings.fontSize}px`;
        });
    }

    updateAutoScroll() {
        this.settings.autoScroll = this.autoScrollToggle.checked;
        this.saveSettings();
    }

    // save/load settings
    saveSettings() {
        localStorage.setItem('chat-settings', JSON.stringify(this.settings));
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('chat-settings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
        }

        // apply to UI
        this.applyTheme();
        if (this.fontSizeSelect) this.fontSizeSelect.value = this.settings.fontSize;
        if (this.autoScrollToggle) this.autoScrollToggle.checked = this.settings.autoScroll;
        this.applyFontSize();
    }

    // clear chat
    clearChat() {
        if (!confirm('Are you sure you want to clear the current chat?')) {
            return;
        }

        const messages = this.chatMessages.querySelectorAll('.message:not(.welcome-message)');
        messages.forEach(msg => msg.remove());
    }

    clearAllData() {
        if (!confirm('Are you sure you want to clear ALL chat data including settings?')) {
            return;
        }

        localStorage.clear();
        
        this.settings = {
            theme: 'light',
            fontSize: 16,
            autoScroll: true
        };
        
        this.loadSettings();
        this.clearChat();
        this.hideSettings();
    }

    // utility functions
    setSendButtonState(enabled) {
        if (this.sendButton) {
            this.sendButton.disabled = !enabled;
        }
    }

    updateCharCount() {
        if (this.charCount && this.userInput) {
            const length = this.userInput.value.length;
            this.charCount.textContent = `${length}/2000`;
            
            if (length > 1800) {
                this.charCount.style.color = 'var(--error-color)';
            } else {
                this.charCount.style.color = 'var(--text-color-muted)';
            }
        }
    }

    adjustTextareaHeight() {
        if (this.userInput) {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
        }
    }

    scrollToBottom() {
        if (this.chatMessages) {
            setTimeout(() => {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }, 100);
        }
    }

    formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    }

    setWelcomeTime() {
        const welcomeTime = document.getElementById('welcome-time');
        if (welcomeTime) {
            welcomeTime.textContent = this.formatTime(new Date());
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    // connection monitoring
    setupConnectionMonitoring() {
        const updateStatus = (isOnline) => {
            const indicator = document.querySelector('.status-indicator');
            if (indicator) {
                indicator.classList.toggle('offline', !isOnline);
            }
        };

        window.addEventListener('online', () => updateStatus(true));
        window.addEventListener('offline', () => updateStatus(false));
        
        updateStatus(navigator.onLine);
    }
}

// initialize
const chatInterface = new ChatInterface();

// error handling
window.addEventListener('error', (e) => console.error('Error:', e.error));
window.addEventListener('unhandledrejection', (e) => console.error('Unhandled rejection:', e.reason));