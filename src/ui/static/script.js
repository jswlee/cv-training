// Beach Conditions Agent - Frontend JavaScript

class BeachConditionsApp {
    constructor() {
        this.apiBaseUrl = '';
        this.isLoading = false;
        this.sessionId = this.generateSessionId();
        
        this.initializeElements();
        this.bindEvents();
        this.checkSystemStatus();
        this.loadInitialConditions();
    }
    
    initializeElements() {
        // Status elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        
        // Condition elements
        this.peopleCount = document.getElementById('peopleCount');
        this.peopleDetail = document.getElementById('peopleDetail');
        this.weatherCondition = document.getElementById('weatherCondition');
        this.weatherDetail = document.getElementById('weatherDetail');
        this.waterActivity = document.getElementById('waterActivity');
        this.waterDetail = document.getElementById('waterDetail');
        
        // Snapshot elements
        this.snapshotImage = document.getElementById('snapshotImage');
        this.snapshotPlaceholder = document.getElementById('snapshotPlaceholder');
        this.refreshSnapshotBtn = document.getElementById('refreshSnapshot');
        
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.quickButtons = document.querySelectorAll('.quick-btn');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }
    
    bindEvents() {
        // Chat input events
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Quick question buttons
        this.quickButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.getAttribute('data-question');
                this.chatInput.value = question;
                this.sendMessage();
            });
        });
        
        // Refresh snapshot button
        this.refreshSnapshotBtn.addEventListener('click', () => this.refreshSnapshot());
        
        // Auto-refresh conditions every 30 seconds
        setInterval(() => this.loadInitialConditions(), 30000);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateStatus('online', 'System Online');
            } else {
                this.updateStatus('offline', 'System Issues');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('offline', 'Connection Error');
        }
    }
    
    updateStatus(status, text) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
    }
    
    async loadInitialConditions() {
        try {
            const response = await fetch('/conditions');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.updateConditionsDisplay(data);
            this.loadSnapshot(data.snapshot_info.snapshot_path);
            
        } catch (error) {
            console.error('Failed to load conditions:', error);
            this.showConditionsError();
        }
    }
    
    updateConditionsDisplay(data) {
        const { people, weather } = data;
        
        // Update people count
        this.peopleCount.textContent = people.total_people;
        this.peopleDetail.textContent = `${people.people_on_beach} on beach, ${people.people_other} other areas`;
        
        // Update weather
        this.weatherCondition.textContent = this.formatWeatherCondition(weather.weather_condition);
        this.weatherDetail.textContent = `${weather.cloud_coverage_percent}% clouds${weather.is_raining ? ', raining' : ''}`;
        
        // Update water activity
        this.waterActivity.textContent = people.people_in_water;
        this.waterDetail.textContent = this.getActivityLevel(people.people_in_water);
    }
    
    formatWeatherCondition(condition) {
        return condition.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    getActivityLevel(count) {
        if (count === 0) return 'No swimmers';
        if (count <= 3) return 'Light activity';
        if (count <= 8) return 'Moderate activity';
        if (count <= 15) return 'High activity';
        return 'Very busy';
    }
    
    showConditionsError() {
        this.peopleCount.textContent = '--';
        this.peopleDetail.textContent = 'Unable to load data';
        this.weatherCondition.textContent = '--';
        this.weatherDetail.textContent = 'Unable to load data';
        this.waterActivity.textContent = '--';
        this.waterDetail.textContent = 'Unable to load data';
    }
    
    async loadSnapshot(snapshotPath) {
        if (!snapshotPath) return;
        
        try {
            // Extract filename from path
            const filename = snapshotPath.split('/').pop();
            const imageUrl = `/image/${filename}`;
            
            // Load image
            const img = new Image();
            img.onload = () => {
                this.snapshotImage.src = imageUrl;
                this.snapshotImage.style.display = 'block';
                this.snapshotPlaceholder.style.display = 'none';
            };
            img.onerror = () => {
                this.showSnapshotError();
            };
            img.src = imageUrl;
            
        } catch (error) {
            console.error('Failed to load snapshot:', error);
            this.showSnapshotError();
        }
    }
    
    showSnapshotError() {
        this.snapshotImage.style.display = 'none';
        this.snapshotPlaceholder.style.display = 'flex';
        this.snapshotPlaceholder.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <p>Unable to load snapshot</p>
        `;
    }
    
    async refreshSnapshot() {
        this.refreshSnapshotBtn.disabled = true;
        this.refreshSnapshotBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Capturing...';
        
        try {
            const response = await fetch('/snapshot', { method: 'POST' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            await this.loadSnapshot(data.snapshot_path);
            
            // Also refresh conditions
            await this.loadInitialConditions();
            
        } catch (error) {
            console.error('Failed to refresh snapshot:', error);
            this.showSnapshotError();
        } finally {
            this.refreshSnapshotBtn.disabled = false;
            this.refreshSnapshotBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Snapshot';
        }
    }
    
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        
        // Show loading state
        this.setLoading(true);
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add bot response to chat
            this.addMessage(data.response, 'bot');
            
            // Update conditions if new analysis was performed
            if (data.analysis_data && Object.keys(data.analysis_data).length > 0) {
                this.updateConditionsFromAnalysis(data.analysis_data);
            }
            
            // Update snapshot if new one was captured
            if (data.snapshot_path) {
                await this.loadSnapshot(data.snapshot_path);
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const textP = document.createElement('p');
        textP.textContent = content;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        contentDiv.appendChild(textP);
        contentDiv.appendChild(timeDiv);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    updateConditionsFromAnalysis(analysisData) {
        if (analysisData.people) {
            const people = analysisData.people;
            this.peopleCount.textContent = people.total_people || 0;
            this.peopleDetail.textContent = `${people.people_on_beach || 0} on beach, ${people.people_other || 0} other areas`;
            this.waterActivity.textContent = people.people_in_water || 0;
            this.waterDetail.textContent = this.getActivityLevel(people.people_in_water || 0);
        }
        
        if (analysisData.weather) {
            const weather = analysisData.weather;
            this.weatherCondition.textContent = this.formatWeatherCondition(weather.weather_condition || 'unknown');
            this.weatherDetail.textContent = `${weather.cloud_coverage_percent || 0}% clouds${weather.is_raining ? ', raining' : ''}`;
        }
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading;
        this.chatInput.disabled = loading;
        
        if (loading) {
            this.loadingOverlay.classList.add('show');
        } else {
            this.loadingOverlay.classList.remove('show');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BeachConditionsApp();
});
