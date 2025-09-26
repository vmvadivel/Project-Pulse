/**
 * Complete functionality for file upload and management
 */

class AdminDashboard {
    constructor() {
        this.apiBase = 'http://127.0.0.1:8000';
        this.maxRetries = 3;
        this.retryDelay = 1000;
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    /**
     * Initialize the dashboard
     */
    init() {
        console.log('Initializing Admin Dashboard...');
        
        try {
            // Get DOM elements
            this.initializeElements();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Load initial data
            this.loadInitialData();
            
            // Start auto-refresh
            this.startAutoRefresh();
            
            console.log('Admin Dashboard initialized successfully');
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
        }
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        // File input and upload area
        this.fileInput = document.getElementById('file-input');
        this.uploadArea = document.getElementById('upload-area');
        
        // Status and progress elements
        this.uploadStatus = document.getElementById('upload-status');
        this.fileStatus = document.getElementById('file-status');
        this.loadingContainer = document.getElementById('loading-container');
        this.progressBar = document.getElementById('progress-bar');
        this.progressFill = document.getElementById('progress-fill');
        
        // File list and buttons
        this.fileList = document.getElementById('file-list');
        this.clearAllBtn = document.getElementById('clear-all-btn');
        
        // Statistics elements
        this.totalFilesEl = document.getElementById('total-files');
        this.totalDocumentsEl = document.getElementById('total-documents');
        this.totalChunksEl = document.getElementById('total-chunks');

        // Validate required elements
        const requiredElements = [
            'fileInput', 'uploadArea', 'uploadStatus', 'fileStatus', 
            'fileList', 'clearAllBtn', 'totalFilesEl', 'totalDocumentsEl', 'totalChunksEl'
        ];

        const missingElements = requiredElements.filter(el => !this[el]);
        if (missingElements.length > 0) {
            console.error('Missing required elements:', missingElements);
            throw new Error('Dashboard initialization failed - missing elements');
        }
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files);
            }
        });

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files);
            }
        });

        // Clear all button
        this.clearAllBtn.addEventListener('click', () => {
            this.clearAllFiles();
        });

        // Prevent default drag behaviors on document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
    }

    /**
     * Load initial data
     */
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadFileList(),
                this.loadStats()
            ]);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showStatus('file', 'Failed to load dashboard data. Please refresh the page.', 'error');
        }
    }

    /**
     * Handle file upload process
     */
    async handleFileUpload(files) {
        if (!files || files.length === 0) {
            console.warn('No files provided for upload');
            return;
        }

        console.log(`Starting upload process for ${files.length} file(s)`);
        
        // Show loading state
        this.showLoading(true);
        this.showProgress(true);
        this.clearStatus('upload');

        let successCount = 0;
        let errorCount = 0;
        const errors = [];

        // Process each file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const progress = ((i + 1) / files.length) * 100;
            
            console.log(`Processing file ${i + 1}/${files.length}: ${file.name}`);
            this.updateProgress(progress);

            try {
                await this.uploadSingleFile(file);
                successCount++;
                console.log(`Successfully uploaded: ${file.name}`);
            } catch (error) {
                errorCount++;
                errors.push({ file: file.name, error: error.message });
                console.error(`Failed to upload ${file.name}:`, error);
            }
        }

        // Hide loading state
        this.showLoading(false);
        this.showProgress(false);

        // Show results
        this.showUploadResults(successCount, errorCount, errors);

        // Refresh data if any uploads succeeded
        if (successCount > 0) {
            await this.loadFileList();
            await this.loadStats();
        }

        // Clear file input
        this.fileInput.value = '';
    }

    /**
     * Upload a single file
     */
    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await this.fetchWithRetry(`${this.apiBase}/ingest`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Show upload results
     */
    showUploadResults(successCount, errorCount, errors) {
        if (successCount > 0 && errorCount === 0) {
            this.showStatus('upload', 
                `Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}!`, 
                'success'
            );
        } else if (successCount > 0 && errorCount > 0) {
            this.showStatus('upload', 
                `Uploaded ${successCount} file${successCount > 1 ? 's' : ''}, failed ${errorCount}. Check console for details.`, 
                'info'
            );
            console.warn('Upload errors:', errors);
        } else {
            const firstError = errors[0] ? errors[0].error : 'Unknown error';
            this.showStatus('upload', 
                `Failed to upload ${errorCount} file${errorCount > 1 ? 's' : ''}: ${firstError}`, 
                'error'
            );
        }
    }

    /**
     * Load file list from server
     */
    async loadFileList() {
        try {
            console.log('Loading file list...');
            const response = await this.fetchWithRetry(`${this.apiBase}/files`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.renderFileList(data.files || []);
            
            console.log(`Loaded ${data.files?.length || 0} files`);
        } catch (error) {
            console.error('Failed to load file list:', error);
            this.showStatus('file', 'Failed to load file list. Please refresh the page.', 'error');
            this.renderFileList([]);
        }
    }

    /**
     * Load statistics from server
     */
    async loadStats() {
        try {
            console.log('Loading statistics...');
            const response = await this.fetchWithRetry(`${this.apiBase}/files`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            
            // Update statistics with animation
            this.animateCounter(this.totalFilesEl, data.total_files || 0);
            this.animateCounter(this.totalDocumentsEl, data.total_documents || 0);
            this.animateCounter(this.totalChunksEl, data.total_chunks || 0);
            
            console.log('Statistics updated');
        } catch (error) {
            console.error('Failed to load statistics:', error);
            // Set to 0 on error
            if (this.totalFilesEl) this.totalFilesEl.textContent = '0';
            if (this.totalDocumentsEl) this.totalDocumentsEl.textContent = '0';
            if (this.totalChunksEl) this.totalChunksEl.textContent = '0';
        }
    }

    /**
     * Render file list in the UI
     */
    renderFileList(files) {
        if (!this.fileList) {
            console.error('File list element not found');
            return;
        }

        if (!files || files.length === 0) {
            this.fileList.innerHTML = `
                <div class="empty-state">
                    <img src="assets/icons/files.svg" alt="No Files" class="empty-state-icon">
                    <p>No files uploaded yet. Upload your first document to get started!</p>
                </div>
            `;
            return;
        }

        const fileListHTML = files.map((file, index) => {
            const animationDelay = `${index * 0.1}s`;
            
            return `
                <div class="file-item" style="animation-delay: ${animationDelay}">
                    <div class="file-info">
                        <div class="file-name">
                            ${this.escapeHtml(file.filename)}
                        </div>
                        <div class="file-details">
                            <span class="file-type-badge">
                                ${file.file_type || 'Unknown'}
                            </span>
                            <span>${file.num_documents || 0} docs</span>
                            <span>${file.num_chunks || 0} chunks</span>
                            <span>${file.file_size_formatted || this.formatFileSize(file.file_size || 0)}</span>
                            ${file.upload_date ? `<span>${this.formatDate(file.upload_date)}</span>` : ''}
                        </div>
                    </div>
                    <button class="delete-btn" onclick="adminDashboard.deleteFile('${this.escapeHtml(file.filename)}')">
                        🗑️ Delete
                    </button>
                </div>
            `;
        }).join('');

        this.fileList.innerHTML = fileListHTML;
        console.log(`Rendered ${files.length} files in list`);
    }

    /**
     * Clear all files
     */
    async clearAllFiles() {
        const fileCount = this.fileList?.querySelectorAll('.file-item').length || 0;
        
        if (fileCount === 0) {
            this.showStatus('file', 'No files to clear.', 'info');
            return;
        }

        if (!confirm(`Are you sure you want to clear all ${fileCount} files? This action cannot be undone.`)) {
            return;
        }

        try {
            console.log('Clearing all files...');
            this.showStatus('file', 'Clearing all files...', 'info');

            const response = await this.fetchWithRetry(`${this.apiBase}/files`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            this.showStatus('file', 'All files cleared successfully!', 'success');
            
            // Refresh data
            await this.loadFileList();
            await this.loadStats();
            
            console.log('All files cleared successfully');
        } catch (error) {
            console.error('Failed to clear files:', error);
            this.showStatus('file', `Failed to clear files: ${error.message}`, 'error');
        }
    }

    /**
     * Delete a specific file (placeholder for future implementation)
     */
    async deleteFile(filename) {
        console.log(`Delete request for file: ${filename}`);
        
        // Since individual file deletion is not implemented in the backend yet
        this.showStatus('file', 
            'Individual file deletion is not yet implemented. Use "Clear All" to remove all files.', 
            'info'
        );
    }

    /**
     * Fetch with retry logic
     */
    async fetchWithRetry(url, options = {}, retries = this.maxRetries) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                },
            });
            return response;
        } catch (error) {
            if (retries > 0) {
                console.warn(`Fetch failed, retrying... (${retries} attempts left)`);
                await this.delay(this.retryDelay);
                return this.fetchWithRetry(url, options, retries - 1);
            }
            throw error;
        }
    }

    /**
     * Show/hide loading spinner
     */
    showLoading(show) {
        if (this.loadingContainer) {
            this.loadingContainer.style.display = show ? 'flex' : 'none';
        }
    }

    /**
     * Show/hide progress bar
     */
    showProgress(show) {
        if (this.progressBar) {
            this.progressBar.style.display = show ? 'block' : 'none';
        }
        if (!show && this.progressFill) {
            this.updateProgress(0);
        }
    }

    /**
     * Update progress bar
     */
    updateProgress(percent) {
        if (this.progressFill) {
            this.progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        }
    }

    /**
     * Show status message
     */
    showStatus(type, message, status) {
        const statusEl = type === 'upload' ? this.uploadStatus : this.fileStatus;
        
        if (!statusEl) {
            console.warn(`Status element not found for type: ${type}`);
            return;
        }

        statusEl.className = `status-message status-${status} show`;
        statusEl.textContent = message;

        // Auto-hide after delay
        setTimeout(() => {
            statusEl.classList.remove('show');
        }, status === 'error' ? 8000 : 5000);
    }

    /**
     * Clear status message
     */
    clearStatus(type) {
        const statusEl = type === 'upload' ? this.uploadStatus : this.fileStatus;
        if (statusEl) {
            statusEl.classList.remove('show');
        }
    }

    /**
     * Animate counter from current value to target
     */
    animateCounter(element, target) {
        if (!element) return;

        const current = parseInt(element.textContent) || 0;
        const increment = Math.ceil((target - current) / 20);
        
        if (increment === 0) {
            element.textContent = target;
            return;
        }

        let value = current;
        const timer = setInterval(() => {
            value += increment;
            
            if ((increment > 0 && value >= target) || (increment < 0 && value <= target)) {
                value = target;
                clearInterval(timer);
            }
            
            element.textContent = value;
        }, 50);
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    }

    /**
     * Format date
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    /**
     * Helper to escape HTML characters
     */
    escapeHtml(unsafe) {
        return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
    
    /**
     * Helper to delay execution
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Auto-refresh logic
     */
    startAutoRefresh() {
        setInterval(() => {
            console.log('Auto-refreshing dashboard data...');
            this.loadStats();
            this.loadFileList();
        }, 30000); // Refresh every 30 seconds
    }
}

// Instantiate the dashboard when the script runs
const adminDashboard = new AdminDashboard();