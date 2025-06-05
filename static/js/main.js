// Main JavaScript for Document Search Application

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeFileUpload();
    initializeSearch();
    initializeFlashMessages();
    initializeTooltips();
});

// File Upload Enhancement
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const dropZone = document.getElementById('dropZone');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');

    if (!fileInput || !dropZone) return;

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        handleFiles(e.target.files);
    });

    // Drag and drop functionality
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        if (!dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('dragover');
        }
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFiles(files);
        }
    });

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!validateUploadForm()) {
                e.preventDefault();
                return false;
            }

            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Uploading...';
            
            // Add timeout to prevent infinite loading
            setTimeout(() => {
                if (submitBtn.disabled) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-upload mr-2"></i>Upload Document';
                }
            }, 30000); // 30 second timeout
        });
    }

    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        
        // Validate file type
        const allowedTypes = ['application/pdf', 'application/msword', 
                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                             'text/plain'];
        const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt'];
        
        const isValidType = allowedTypes.includes(file.type) || 
                           allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValidType) {
            showError('Please select a valid file type (PDF, DOC, DOCX, or TXT)');
            fileInput.value = '';
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showError('File size must be less than 16MB');
            fileInput.value = '';
            return;
        }

        // Display file info
        fileName.textContent = file.name;
        fileSize.textContent = `(${formatFileSize(file.size)})`;
        fileInfo.classList.remove('hidden');
    }

    function validateUploadForm() {
        const file = fileInput.files[0];
        const description = document.getElementById('description').value.trim();

        if (!file) {
            showError('Please select a file to upload');
            return false;
        }

        if (!description) {
            showError('Please provide a description for the document');
            document.getElementById('description').focus();
            return false;
        }

        if (description.length < 10) {
            showError('Please provide a more detailed description (at least 10 characters)');
            document.getElementById('description').focus();
            return false;
        }

        return true;
    }
}

// Search Enhancement
function initializeSearch() {
    const searchForm = document.getElementById('searchForm');
    const searchBtn = document.getElementById('searchBtn');
    const queryInput = document.getElementById('query');
    const searchExamples = document.querySelectorAll('.search-example');

    // Handle search form submission
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const query = queryInput.value.trim();
            
            if (!query) {
                e.preventDefault();
                showError('Please enter a search query');
                queryInput.focus();
                return false;
            }

            if (query.length < 3) {
                e.preventDefault();
                showError('Search query must be at least 3 characters long');
                queryInput.focus();
                return false;
            }

            // Show loading state
            searchBtn.disabled = true;
            searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';
            
            // Add timeout to prevent infinite loading
            setTimeout(() => {
                if (searchBtn.disabled) {
                    searchBtn.disabled = false;
                    searchBtn.innerHTML = '<i class="fas fa-search mr-2"></i>Search';
                }
            }, 30000); // 30 second timeout
        });
    }

    // Handle search example clicks
    searchExamples.forEach(example => {
        example.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            if (queryInput) {
                queryInput.value = query;
                queryInput.focus();
            }
        });
    });

    // Add search suggestions (debounced)
    if (queryInput) {
        let searchTimeout;
        queryInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                // Could add search suggestions here
                validateSearchInput();
            }, 300);
        });
    }

    function validateSearchInput() {
        const query = queryInput.value.trim();
        const isValid = query.length >= 3;
        
        if (searchBtn) {
            searchBtn.disabled = !isValid;
        }
    }
}

// Flash Messages Enhancement
function initializeFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');
    
    flashMessages.forEach(message => {
        // Auto-hide success messages after 5 seconds
        if (message.classList.contains('flash-success')) {
            setTimeout(() => {
                fadeOut(message);
            }, 5000);
        }

        // Add close button functionality
        const closeBtn = message.querySelector('button');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                fadeOut(message);
            });
        }
    });
}

// Tooltip Enhancement
function initializeTooltips() {
    const elements = document.querySelectorAll('[data-tooltip]');
    
    elements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
    // Create and show error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'flash-message flash-error fade-in';
    errorDiv.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-exclamation-circle mr-2"></i>
            <span>${message}</span>
        </div>
        <button onclick="this.parentElement.style.display='none'" class="ml-auto">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Insert at the top of the main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(errorDiv, main.firstChild);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            fadeOut(errorDiv);
        }, 5000);
    }
}

function showSuccess(message) {
    // Create and show success message
    const successDiv = document.createElement('div');
    successDiv.className = 'flash-message flash-success fade-in';
    successDiv.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-check-circle mr-2"></i>
            <span>${message}</span>
        </div>
        <button onclick="this.parentElement.style.display='none'" class="ml-auto">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Insert at the top of the main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(successDiv, main.firstChild);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            fadeOut(successDiv);
        }, 5000);
    }
}

function fadeOut(element) {
    element.style.opacity = '0';
    element.style.transform = 'translateY(-10px)';
    element.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    
    setTimeout(() => {
        if (element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }, 300);
}

function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip show';
    tooltip.textContent = e.target.getAttribute('data-tooltip');
    
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
    
    e.target._tooltip = tooltip;
}

function hideTooltip(e) {
    if (e.target._tooltip) {
        document.body.removeChild(e.target._tooltip);
        delete e.target._tooltip;
    }
}

// File Download Enhancement
function downloadFile(filename) {
    // Add loading state to download button
    const downloadBtn = event.target.closest('a');
    if (downloadBtn) {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Downloading...';
        
        // Reset after 3 seconds
        setTimeout(() => {
            downloadBtn.innerHTML = originalText;
        }, 3000);
    }
}

// Keyboard Shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + / to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        const searchInput = document.getElementById('query');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        const searchInput = document.getElementById('query');
        if (searchInput && document.activeElement === searchInput) {
            searchInput.blur();
        }
    }
});

// Performance Monitoring
function trackPerformance(action, startTime) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    console.log(`${action} completed in ${duration.toFixed(2)}ms`);
    
    // Could send to analytics service
    if (duration > 1000) {
        console.warn(`Slow operation detected: ${action} took ${duration.toFixed(2)}ms`);
    }
}

// Error Handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // Could send to error tracking service
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    // Could send to error tracking service
});
