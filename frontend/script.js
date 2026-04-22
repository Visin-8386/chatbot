/**
 * DocSearch — Frontend Logic
 * Handles file upload, search, document management
 */

const API_BASE = '';

// === DOM Elements ===
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchHero = document.getElementById('searchHero');
const resultsList = document.getElementById('resultsList');
const loading = document.getElementById('loading');
const noResults = document.getElementById('noResults');
const aiAnswerContainer = document.getElementById('aiAnswerContainer');
const aiAnswerContent = document.getElementById('aiAnswerContent');
const aiCitations = document.getElementById('aiCitations');
const resultsHeader = document.getElementById('resultsHeader');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFilename = document.getElementById('progressFilename');
const progressStatus = document.getElementById('progressStatus');
const progressFill = document.getElementById('progressFill');
const documentsList = document.getElementById('documentsList');
const emptyDocs = document.getElementById('emptyDocs');
const docCount = document.getElementById('docCount');
const statDocs = document.getElementById('statDocs');
const statChunks = document.getElementById('statChunks');
const toastContainer = document.getElementById('toastContainer');
const sidebar = document.getElementById('sidebar');
const menuBtn = document.getElementById('menuBtn');
const sidebarToggle = document.getElementById('sidebarToggle');

// === Toast Notifications ===
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

// === Search ===
let searchTimeout = null;

async function performSearch(query) {
    if (!query.trim()) return;

    // Minimize hero
    searchHero.classList.add('minimized');

    // Show loading
    loading.classList.remove('hidden');
    const loadingText = loading.querySelector('p');
    if (loadingText) loadingText.textContent = 'Đang tìm kiếm tài liệu...';
    noResults.classList.add('hidden');
    aiAnswerContainer.classList.add('hidden');
    resultsHeader.classList.add('hidden');
    resultsList.innerHTML = '';

    try {
        // Update loading message for AI generation
        setTimeout(() => {
            if (!loading.classList.contains('hidden') && loadingText) {
                loadingText.textContent = 'Đang tạo câu trả lời AI...';
            }
        }, 1500);

        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query.trim(), top_k: 5 })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Search failed');
        }

        const data = await response.json();
        loading.classList.add('hidden');

        if (data.results.length === 0) {
            noResults.classList.remove('hidden');
            return;
        }

        // Render AI Answer if available
        if (data.ai_answer) {
            aiAnswerContainer.classList.remove('hidden');
            // Format basic markdown (bold and newlines)
            let formattedAnswer = escapeHtml(data.ai_answer)
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            aiAnswerContent.innerHTML = formattedAnswer;

            // Render citations from ai_sources
            renderCitations(data.ai_sources || [], data.results);
        }

        // Show sources header
        resultsHeader.classList.remove('hidden');
        renderResults(data.results, query.trim());
    } catch (error) {
        loading.classList.add('hidden');
        showToast(`Lỗi tìm kiếm: ${error.message}`, 'error');
    }
}

/**
 * Render citations block below the AI answer
 */
function renderCitations(aiSources, results) {
    if (!aiSources || aiSources.length === 0) {
        aiCitations.innerHTML = '';
        return;
    }

    // Build unique sources from search results metadata
    const sourcesMap = new Map();
    results.forEach((res) => {
        const source = res.metadata.source || 'Unknown';
        const key = `${source}|${res.metadata.page || ''}|${res.metadata.sheet || ''}`;
        if (!sourcesMap.has(key)) {
            sourcesMap.set(key, {
                file: source,
                page: res.metadata.page,
                sheet: res.metadata.sheet,
                similarity: res.similarity
            });
        }
    });

    const sources = Array.from(sourcesMap.values());
    const ext = (filename) => {
        const e = filename.split('.').pop().toLowerCase();
        return ['pdf', 'docx', 'xlsx', 'txt'].includes(e) ? e : 'txt';
    };

    let html = '<div class="citations-header">';
    html += '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">';
    html += '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>';
    html += '<polyline points="14 2 14 8 20 8"></polyline></svg>';
    html += '<span>Nguồn tham khảo</span>';
    html += '</div>';
    html += '<div class="citations-list">';

    sources.forEach((src) => {
        const fileExt = ext(src.file);
        let location = '';
        if (src.page) location += `Trang ${src.page}`;
        if (src.sheet) location += `${location ? ' · ' : ''}Sheet: ${src.sheet}`;

        html += `
            <div class="citation-chip">
                <span class="citation-icon ${fileExt}">${fileExt}</span>
                <div class="citation-info">
                    <span class="citation-file">${escapeHtml(src.file)}</span>
                    ${location ? `<span class="citation-location">${location}</span>` : ''}
                </div>
                <span class="citation-score">${src.similarity}%</span>
            </div>
        `;
    });

    html += '</div>';
    aiCitations.innerHTML = html;
}

function renderResults(results, query) {
    resultsList.innerHTML = '';

    results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.animationDelay = `${index * 0.08}s`;

        const ext = getFileExtension(result.metadata.source || '');
        const scoreClass = result.similarity >= 70 ? 'high' : result.similarity >= 50 ? 'medium' : 'low';

        // Build meta info
        let metaParts = [];
        if (result.metadata.page) metaParts.push(`Trang ${result.metadata.page}`);
        if (result.metadata.sheet) metaParts.push(`Sheet: ${result.metadata.sheet}`);
        if (result.metadata.chunk_index !== undefined) metaParts.push(`Đoạn ${parseInt(result.metadata.chunk_index) + 1}`);
        const metaText = metaParts.join(' · ');

        // Highlight query terms in result text
        const highlightedText = highlightText(result.text, query);

        card.innerHTML = `
            <div class="result-header">
                <div class="result-source">
                    <span class="result-index">${index + 1}</span>
                    <div class="result-file-icon ${ext}">${ext}</div>
                    <div>
                        <div class="result-filename">${escapeHtml(result.metadata.source || 'Unknown')}</div>
                        <div class="result-meta">${metaText}</div>
                    </div>
                </div>
                <div class="result-score">
                    <div class="score-bar">
                        <div class="score-fill ${scoreClass}" style="width: ${result.similarity}%"></div>
                    </div>
                    <span class="score-text ${scoreClass}">${result.similarity}%</span>
                </div>
            </div>
            <div class="result-text">${highlightedText}</div>
        `;

        resultsList.appendChild(card);
    });
}

function highlightText(text, query) {
    const escaped = escapeHtml(text);
    const words = query.split(/\s+/).filter(w => w.length > 1);
    
    if (words.length === 0) return escaped;

    // Create regex to match any query word
    const regex = new RegExp(
        `(${words.map(w => escapeRegex(w)).join('|')})`,
        'gi'
    );

    return escaped.replace(regex, '<mark>$1</mark>');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getFileExtension(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return ['pdf', 'docx', 'xlsx', 'txt'].includes(ext) ? ext : 'txt';
}

// === File Upload ===
async function uploadFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const supported = ['.pdf', '.docx', '.xlsx', '.txt'];

    if (!supported.includes(ext)) {
        showToast(`Không hỗ trợ file ${ext}. Chỉ hỗ trợ: ${supported.join(', ')}`, 'error');
        return;
    }

    // Show progress
    uploadProgress.classList.remove('hidden');
    progressFilename.textContent = file.name;
    progressStatus.textContent = 'Đang tải lên...';
    progressFill.style.width = '20%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressFill.style.width = '50%';
        progressStatus.textContent = 'Đang xử lý tài liệu...';

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        progressFill.style.width = '90%';

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();
        progressFill.style.width = '100%';
        progressStatus.textContent = 'Hoàn tất!';

        showToast(`✓ Đã xử lý "${file.name}" → ${data.chunks} đoạn văn`, 'success');

        // Refresh documents list
        await loadDocuments();
    } catch (error) {
        showToast(`Lỗi: ${error.message}`, 'error');
        progressStatus.textContent = 'Thất bại!';
    } finally {
        setTimeout(() => {
            uploadProgress.classList.add('hidden');
            progressFill.style.width = '0%';
        }, 1500);
    }
}

// === Documents Management ===
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();

        // Update stats
        statDocs.textContent = data.total_documents;
        statChunks.textContent = data.total_chunks;
        docCount.textContent = data.total_documents;

        // Render document list
        if (data.documents.length === 0) {
            emptyDocs.classList.remove('hidden');
            // Clear previous doc items
            const items = documentsList.querySelectorAll('.doc-item');
            items.forEach(item => item.remove());
            return;
        }

        emptyDocs.classList.add('hidden');

        // Clear and re-render
        const items = documentsList.querySelectorAll('.doc-item');
        items.forEach(item => item.remove());

        data.documents.forEach(doc => {
            const ext = getFileExtension(doc.source);
            const item = document.createElement('div');
            item.className = 'doc-item';
            item.innerHTML = `
                <div class="doc-icon ${ext}">${ext}</div>
                <div class="doc-info">
                    <div class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</div>
                    <div class="doc-chunks">${doc.chunk_count} đoạn văn</div>
                </div>
                <button class="doc-delete" data-id="${doc.doc_id}" title="Xóa tài liệu" aria-label="Delete document">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            `;
            documentsList.appendChild(item);
        });
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Bạn có chắc muốn xóa tài liệu này?')) return;

    try {
        const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Delete failed');
        }

        showToast('Đã xóa tài liệu', 'success');
        await loadDocuments();
    } catch (error) {
        showToast(`Lỗi: ${error.message}`, 'error');
    }
}

// === Event Listeners ===

// Search
searchBtn.addEventListener('click', () => performSearch(searchInput.value));
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performSearch(searchInput.value);
});

// Upload - Click
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => uploadFile(file));
    fileInput.value = '';
});

// Upload - Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => uploadFile(file));
});

// Document delete
documentsList.addEventListener('click', (e) => {
    const deleteBtn = e.target.closest('.doc-delete');
    if (deleteBtn) {
        deleteDocument(deleteBtn.dataset.id);
    }
});

// Mobile sidebar
menuBtn.addEventListener('click', () => {
    sidebar.classList.add('open');
    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    overlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        overlay.remove();
    });
    document.body.appendChild(overlay);
});

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.remove('open');
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.remove();
});

// === Init ===
loadDocuments();
