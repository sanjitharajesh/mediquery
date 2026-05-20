// MediQuery Frontend JavaScript

// API base URL - empty string uses relative paths (works when the frontend is served by the
// FastAPI backend, both locally and on Render). Set this to your full Render service URL
// (e.g. 'https://mediquery.onrender.com') only if you host the frontend separately.
const API_BASE_URL = '';

// DOM Elements
const questionInput = document.getElementById('question-input');
const getAnswerBtn = document.getElementById('get-answer-btn');
const clearBtn = document.getElementById('clear-btn');
const answerBox = document.getElementById('answer-box');
const chipButtons = document.querySelectorAll('.chip-btn');

// State
let isLoading = false;
let loadingStepInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupSectionNavigation();
    setupEventListeners();
});

// ------------------------------------------------------------------ //
// Section Navigation
// ------------------------------------------------------------------ //
function showSection(id) {
    ['section-landing', 'section-personal', 'section-app'].forEach(sId => {
        const el = document.getElementById(sId);
        if (el) el.style.display = sId === id ? 'flex' : 'none';
    });
}

function getSelectedText(selectId) {
    const select = document.getElementById(selectId);
    if (!select || select.selectedIndex <= 0) return 'Not specified';
    return select.options[select.selectedIndex].text;
}

function populateProfile() {
    const fields = [
        ['age-group',   'profile-age'],
        ['gender',      'profile-gender'],
        ['conditions',  'profile-conditions'],
        ['medications', 'profile-medications'],
    ];
    fields.forEach(([selectId, displayId]) => {
        const el = document.getElementById(displayId);
        if (el) el.textContent = getSelectedText(selectId);
    });
}

function buildProfileContext() {
    const fieldIds = ['age-group', 'gender', 'conditions', 'medications'];
    const parts = fieldIds.map(id => {
        const sel = document.getElementById(id);
        if (!sel || sel.selectedIndex <= 0) return null;
        return sel.options[sel.selectedIndex].text;
    }).filter(Boolean);
    return parts.length ? `Patient context: ${parts.join(', ')}.` : '';
}

function setupSectionNavigation() {
    document.getElementById('btn-get-started')?.addEventListener('click', () => {
        showSection('section-personal');
    });

    document.getElementById('btn-continue')?.addEventListener('click', () => {
        populateProfile();
        showSection('section-app');
    });

    document.getElementById('btn-skip')?.addEventListener('click', (e) => {
        e.preventDefault();
        showSection('section-app');
    });

    document.getElementById('btn-edit-profile')?.addEventListener('click', () => {
        showSection('section-personal');
    });
}

// ------------------------------------------------------------------ //
// Event Listeners
// ------------------------------------------------------------------ //
function setupEventListeners() {
    getAnswerBtn.addEventListener('click', handleGetAnswer);

    clearBtn.addEventListener('click', handleClear);

    chipButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            questionInput.value = question;
            questionInput.focus();
        });
    });

    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isLoading) {
                handleGetAnswer();
            }
        }
    });
}

// ------------------------------------------------------------------ //
// API Logic
// ------------------------------------------------------------------ //
async function handleGetAnswer() {
    const question = questionInput.value.trim();

    if (!question) {
        showError('Please enter a question');
        return;
    }

    if (isLoading) {
        return;
    }

    const context = buildProfileContext();
    const payload = context ? `${context} Question: ${question}` : question;

    setLoading(true);
    showLoadingMessage();

    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: payload }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get answer');
        }

        const data = await response.json();
        displayAnswer(data.answer);
    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

function handleClear() {
    questionInput.value = '';
    answerBox.innerHTML = '<span class="answer-placeholder">Your answer will appear here...</span>';
    answerBox.classList.remove('loading');
    questionInput.focus();
}

function displayAnswer(answer) {
    answerBox.innerHTML = marked.parse(answer);
    answerBox.classList.remove('loading');

    answerBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showLoadingMessage() {
    const steps = [
        'Retrieving FDA documents...',
        'Ranking results with hybrid search...',
        'Generating answer...',
    ];
    let stepIndex = 0;

    const render = (text) => {
        answerBox.innerHTML = `
            <div class="loading-steps">
                <span class="loading-text">${text}</span>
                <span class="loading-dots"><span></span><span></span><span></span></span>
            </div>`;
        answerBox.classList.add('loading');
    };

    render(steps[0]);
    loadingStepInterval = setInterval(() => {
        stepIndex = (stepIndex + 1) % steps.length;
        render(steps[stepIndex]);
    }, 2000);
}

function showError(message) {
    answerBox.innerHTML = `<span style="color:#EF4444;font-size:0.875rem">${message}</span>`;
    answerBox.classList.remove('loading');
}

function setLoading(loading) {
    isLoading = loading;
    getAnswerBtn.disabled = loading;
    getAnswerBtn.textContent = loading ? 'Loading...' : 'Get Answer';

    if (!loading && loadingStepInterval) {
        clearInterval(loadingStepInterval);
        loadingStepInterval = null;
    }
}
