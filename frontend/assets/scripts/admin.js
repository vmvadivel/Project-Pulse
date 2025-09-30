/**
 * Enhanced Admin Dashboard JavaScript
 * Updated to work with the enhanced main.py backend
 * Features: Single file delete, performance monitoring, better error handling
 */

class AdminDashboard {
    constructor() {
        this.apiBase = 'http://127.0.0.1:8000';
        this.maxRetries = 3;
        this.retryDelay = 1000;
        
        // NEW: Define standard and upload timeouts (in milliseconds)
        this.STANDARD_TIMEOUT_MS = 30000; // 30 seconds for fast operations
        this.UPLOAD_TIMEOUT_MS = 180000;  // 3 minutes (180 seconds) for file ingestion

        this.performanceStats = {
            totalUploads: 0,
            totalUploadTime: 0,
            averageUploadTime: 0
        };
        
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
        console.log('Initializing Enhanced Admin Dashboard...');
        
        try {
            this.initializeElements();
            this.setupEventListeners();
            this.loadInitialData();
            this.startAutoRefresh();
            this.checkBackendHealth();
            
            console.log('Enhanced Admin Dashboard initialized successfully');
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

        // File input change with file size validation
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                if (this.validateFiles(e.target.files)) {
                    this.handleFileUpload(e.target.files);
                }
            }
        });

        // Enhanced drag and drop with file validation
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
                if (this.validateFiles(e.dataTransfer.files)) {
                    this.handleFileUpload(e.dataTransfer.files);
                }
            }
        });

        // Clear all button
        this.clearAllBtn.addEventListener('click', () => {
            this.clearAllFiles();
        });

        // Prevent default drag behaviors on document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+U for upload
            if (e.ctrlKey && e.key === 'u') {
                e.preventDefault();
                this.fileInput.click();
            }
            // Ctrl+R for refresh
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                this.loadInitialData();
            }
        });
    }

    /**
     * Validate files before upload
     */
    validateFiles(files) {
        const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
        const invalidFiles = [];

        for (let file of files) {
            if (file.size > MAX_FILE_SIZE) {
                invalidFiles.push(`${file.name} (${this.formatFileSize(file.size)})`);
            }
        }

        if (invalidFiles.length > 0) {
            this.showStatus('upload', 
                `File(s) too large: ${invalidFiles.join(', ')}. Maximum size: 100MB`, 
                'error'
            );
            return false;
        }

        return true;
    }

    /**
     * Check backend health
     */
    async checkBackendHealth() {
        try {
            // Uses the default (standard) timeout
            const response = await this.fetchWithRetry(`${this.apiBase}/health`); 
            if (response.ok) {
                const healthData = await response.json();
                console.log('Backend health check passed:', healthData);
                this.updateConnectionStatus(true);
            } else {
                throw new Error('Health check failed');
            }
        } catch (error) {
            console.warn('Backend health check failed:', error);
            this.updateConnectionStatus(false);
            this.showStatus('file', 'Backend connection issues detected. Some features may not work properly.', 'error');
        }
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(isConnected) {
        // Add a connection status indicator to the header if it doesn't exist
        let statusIndicator = document.getElementById('connection-status');
        if (!statusIndicator) {
            statusIndicator = document.createElement('div');
            statusIndicator.id = 'connection-status';
            statusIndicator.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                transition: all 0.3s ease;
                z-index: 10;
            `;
            // NOTE: Assumes an element with class 'admin-header' exists in your HTML
            const adminHeader = document.querySelector('.admin-header');
            if (adminHeader) {
                adminHeader.appendChild(statusIndicator);
            } else {
                // Fallback for demonstration if admin-header isn't present
                console.warn("Could not find '.admin-header' element to attach status indicator.");
            }
        }

        if (isConnected) {
            statusIndicator.style.backgroundColor = '#10b981';
            statusIndicator.title = 'Connected to backend';
        } else {
            statusIndicator.style.backgroundColor = '#ef4444';
            statusIndicator.title = 'Backend connection lost';
        }
    }

    /**
     * Load initial data with enhanced error handling
     */
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadFileList(),
                this.loadStats(),
                this.loadSystemStats()
            ]);
            this.updateConnectionStatus(true);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateConnectionStatus(false);
            this.showStatus('file', 'Failed to load dashboard data. Please check your connection.', 'error');
        }
    }

    /**
     * Load system statistics from enhanced backend
     */
    async loadSystemStats() {
        try {
            // Uses the default (standard) timeout
            const response = await this.fetchWithRetry(`${this.apiBase}/stats`); 
            if (!response.ok) return;

            const stats = await response.json();
            console.log('System stats loaded:', stats);
            
            // Could add more detailed stats display here if needed
        } catch (error) {
            console.warn('Failed to load system stats:', error);
        }
    }

    /**
     * Enhanced file upload with performance tracking
     */
    async handleFileUpload(files) {
        if (!files || files.length === 0) return;

        const startTime = Date.now();
        console.log(`Starting upload process for ${files.length} file(s)`);
        
        // Show loading state
        this.showLoading(true);
        this.showProgress(true);
        this.clearStatus('upload');

        let successCount = 0;
        let errorCount = 0;
        const errors = [];
        const uploadResults = [];

        // Process each file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const progress = ((i + 1) / files.length) * 100;
            
            console.log(`📄 Processing file ${i + 1}/${files.length}: ${file.name} (${this.formatFileSize(file.size)})`);
            this.updateProgress(progress);

            try {
                // Pass a flag to use the longer timeout
                const result = await this.uploadSingleFile(file); 
                successCount++;
                uploadResults.push(result);
                console.log(`Successfully uploaded: ${file.name} in ${result.processing_time}`);
            } catch (error) {
                errorCount++;
                errors.push({ file: file.name, error: error.message });
                console.error(`Failed to upload ${file.name}:`, error);
            }
        }

        // Update performance stats
        const totalTime = Date.now() - startTime;
        this.updatePerformanceStats(successCount, totalTime);

        // Hide loading state
        this.showLoading(false);
        this.showProgress(false);

        // Show enhanced results
        this.showUploadResults(successCount, errorCount, errors, uploadResults);

        // Refresh data if any uploads succeeded
        if (successCount > 0) {
            await this.loadFileList();
            await this.loadStats();
        }

        this.fileInput.value = '';
    }

    /**
     * Update performance statistics
     */
    updatePerformanceStats(successCount, totalTime) {
        this.performanceStats.totalUploads += successCount;
        this.performanceStats.totalUploadTime += totalTime;
        this.performanceStats.averageUploadTime = 
            this.performanceStats.totalUploadTime / Math.max(this.performanceStats.totalUploads, 1);
        
        console.log('📈 Performance stats updated:', this.performanceStats);
    }

    /**
     * Enhanced upload results display
     */
    showUploadResults(successCount, errorCount, errors, uploadResults) {
        if (successCount > 0 && errorCount === 0) {
            const avgProcessingTime = uploadResults.reduce((sum, result) => 
                sum + parseFloat(result.processing_time), 0) / uploadResults.length;
            
            this.showStatus('upload', 
                `Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}! ` +
                `Average processing time: ${avgProcessingTime.toFixed(2)}s`, 
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
     * Enhanced file upload with better error handling
     */
    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        // IMPORTANT: Call fetchWithRetry with a specific option to use the UPLOAD_TIMEOUT_MS
        const response = await this.fetchWithRetry(`${this.apiBase}/ingest`, {
            method: 'POST',
            body: formData,
        }, null, this.UPLOAD_TIMEOUT_MS); // Pass custom timeout here

        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch {
                errorData = { detail: `HTTP ${response.status}: ${response.statusText}` };
            }
            
            // Handle specific error cases
            if (response.status === 413) {
                throw new Error('File too large (max 100MB)');
            } else if (response.status === 400 && errorData.detail?.includes('already been uploaded')) {
                throw new Error('File already exists');
            } else {
                throw new Error(errorData.detail || `Upload failed: ${response.status}`);
            }
        }

        return await response.json();
    }

    /**
     * Enhanced file list rendering with delete buttons
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
            const processingTime = file.processing_time || 'Unknown';
            
            return `
                <div class="file-item" style="animation-delay: ${animationDelay}">
                    <div class="file-info">
                        <div class="file-name" title="${this.escapeHtml(file.filename)}">
                            <img src="assets/icons/files.svg" alt="File" class="file-type-icon" style="width: 16px; height: 16px; margin-right: 8px;">
                            ${this.escapeHtml(file.filename)}
                        </div>
                        <div class="file-details">
                            <span class="file-type-badge" style="background: ${this.getFileTypeColor(file.file_type)}">
                                ${file.file_type || 'Unknown'}
                            </span>
                            <span title="Documents">${file.num_documents || 0} docs</span>
                            <span title="Text chunks">${file.num_chunks || 0} chunks</span>
                            <span title="File size">${file.file_size_formatted || this.formatFileSize(file.file_size || 0)}</span>
                            <span title="Processing time">${processingTime}</span>
                            ${file.upload_date ? `<span title="Upload date">${this.formatDate(file.upload_date)}</span>` : ''}
                        </div>
                    </div>
                    <div class="file-actions">
                        <button class="delete-btn" onclick="adminDashboard.deleteFile('${this.escapeHtml(file.filename)}')" title="Delete this file">
                            <img src="assets/icons/trash.svg" alt="Delete" class="delete-btn-icon" style="width: 14px; height: 14px; margin-right: 4px;">
                            Delete
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        this.fileList.innerHTML = fileListHTML;
        console.log(`Rendered ${files.length} files in list`);
    }

    /**
     * Get color for file type badge
     */
    getFileTypeColor(fileType) {
        const colorMap = {
            'PDF': '#e53e3e',
            'Word': '#2b77d9',
            'Excel': '#10b981',
            'PowerPoint': '#f59e0b',
            'Text': '#6b7280',
            'CSV': '#8b5cf6',
            'HTML': '#f97316',
            'Markdown': '#06b6d4',
            'JSON': '#eab308'
        };
        return colorMap[fileType] || '#5a67d8';
    }

    /**
     * ENHANCED: Delete specific file using new backend endpoint
     */
    async deleteFile(filename) {
        console.log(`🗑️ Delete request for file: ${filename}`);
        
        if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
            return;
        }

        try {
            this.showStatus('file', `Deleting ${filename}...`, 'info');

            // Uses the default (standard) timeout
            const response = await this.fetchWithRetry(`${this.apiBase}/files/${encodeURIComponent(filename)}`, { 
                method: 'DELETE',
            });

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error('File not found in knowledge base');
                } else {
                    const errorData = await response.json().catch(() => ({ detail: 'Delete failed' }));
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
                }
            }

            const result = await response.json();
            this.showStatus('file', `${result.message}`, 'success');
            
            // Refresh data
            await this.loadFileList();
            await this.loadStats();
            
            console.log(`Successfully deleted: ${filename}`);
        } catch (error) {
            console.error(`Failed to delete ${filename}:`, error);
            this.showStatus('file', `Failed to delete ${filename}: ${error.message}`, 'error');
        }
    }

    /**
     * Load file list with enhanced error handling
     */
    async loadFileList() {
        try {
            console.log('📋 Loading file list...');
            // Uses the default (standard) timeout
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
     * Load statistics with enhanced display
     */
    async loadStats() {
        try {
            console.log('📊 Loading statistics...');
            // Uses the default (standard) timeout
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
     * Enhanced clear all files with better confirmation
     */
    async clearAllFiles() {
        const fileCount = this.fileList?.querySelectorAll('.file-item').length || 0;
        
        if (fileCount === 0) {
            this.showStatus('file', 'No files to clear.', 'info');
            return;
        }

        const confirmMessage = `Are you sure you want to clear all ${fileCount} files?\n\n` +
                                          `This will:\n` +
                                          `• Delete all uploaded documents\n` +
                                          `• Clear the knowledge base\n` +
                                          `• Reset conversation history\n\n` +
                                          `This action cannot be undone.`;

        if (!confirm(confirmMessage)) {
            return;
        }

        try {
            console.log('🗑️ Clearing all files...');
            this.showStatus('file', 'Clearing all files...', 'info');

            // Uses the default (standard) timeout
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
     * MODIFIED: Enhanced fetch with dynamic timeout based on call
     */
    async fetchWithRetry(url, options = {}, retries = this.maxRetries, customTimeout = this.STANDARD_TIMEOUT_MS) {
        
        // Use the custom timeout provided, or fall back to the standard one
        const timeoutMs = customTimeout; 
        
        try {
            const controller = new AbortController();
            // Pass the dynamic timeout value here
            const timeoutId = setTimeout(() => { 
                console.warn(`Request to ${url} timed out after ${timeoutMs / 1000}s`);
                controller.abort();
            }, timeoutMs); 

            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    ...options.headers,
                },
            });

            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            // Only retry if it's NOT an AbortError (timeout)
            if (retries > 0 && error.name !== 'AbortError') { 
                console.warn(`Fetch failed, retrying... (${retries} attempts left):`, error.message);
                await this.delay(this.retryDelay);
                // Pass the customTimeout to the recursive call
                return this.fetchWithRetry(url, options, retries - 1, customTimeout); 
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
     * Enhanced status messages with icons
     */
    showStatus(type, message, status) {
        const statusEl = type === 'upload' ? this.uploadStatus : this.fileStatus;
        
        if (!statusEl) {
            console.warn(`Status element not found for type: ${type}`);
            return;
        }

        statusEl.className = `status-message status-${status} show`;
        statusEl.textContent = `${message}`;

        // Auto-hide after delay
        setTimeout(() => {
            statusEl.classList.remove('show');
        }, status === 'error' ? 10000 : 6000);
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
     * Animate counter with better performance
     */
    animateCounter(element, target) {
        if (!element) return;

        const current = parseInt(element.textContent) || 0;
        const increment = Math.ceil((target - current) / 15); // Faster animation
        
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
        }, 30); // Smoother animation
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
     * Format date with better formatting
     */
    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return dateString;
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    /**
     * Delay utility
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Enhanced auto-refresh with health checks
     */
    startAutoRefresh() {
        setInterval(async () => {
            try {
                console.log('Auto-refreshing dashboard data...');
                await this.loadStats();
                await this.checkBackendHealth();
                // Only refresh file list if there are files
                if (this.fileList?.querySelectorAll('.file-item').length > 0) {
                    await this.loadFileList();
                }
            } catch (error) {
                console.warn('Auto-refresh failed:', error);
                this.updateConnectionStatus(false);
            }
        }, 30000); // Refresh every 30 seconds
    }
}

// Initialize the enhanced admin dashboard
const adminDashboard = new AdminDashboard();

// Global error handlers
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

console.log('Enhanced Admin Dashboard script loaded successfully');