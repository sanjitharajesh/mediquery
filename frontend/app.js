// MediQuery Frontend JavaScript

// DOM Elements
const questionInput = document.getElementById('question-input');
const getAnswerBtn = document.getElementById('get-answer-btn');
const clearBtn = document.getElementById('clear-btn');
const answerBox = document.getElementById('answer-box');
const chipButtons = document.querySelectorAll('.chip-btn');

// State
let isLoading = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('MediQuery initialized');
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Get Answer button
    getAnswerBtn.addEventListener('click', handleGetAnswer);

    // Clear button
    clearBtn.addEventListener('click', handleClear);

    // Common question chips
    chipButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            questionInput.value = question;
            questionInput.focus();
        });
    });

    // Enter key to submit
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isLoading) {
                handleGetAnswer();
            }
        }
    });
}

// Handle Get Answer
async function handleGetAnswer() {
    const question = questionInput.value.trim();

    if (!question) {
        showError('Please enter a question');
        return;
    }

    if (isLoading) {
        return;
    }

    setLoading(true);
    showLoadingMessage();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
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

// Handle Clear
function handleClear() {
    questionInput.value = '';
    answerBox.textContent = 'Your answer will appear here...';
    answerBox.classList.remove('loading');
    questionInput.focus();
}

// Display Answer
function displayAnswer(answer) {
    answerBox.textContent = answer;
    answerBox.classList.remove('loading');

    // Scroll to answer
    answerBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show Loading Message
function showLoadingMessage() {
    answerBox.textContent = 'Searching FDA documents...';
    answerBox.classList.add('loading');
}

// Show Error
function showError(message) {
    answerBox.textContent = message;
    answerBox.classList.remove('loading');
}

// Set Loading State
function setLoading(loading) {
    isLoading = loading;
    getAnswerBtn.disabled = loading;

    if (loading) {
        getAnswerBtn.textContent = 'Loading...';
    } else {
        getAnswerBtn.textContent = 'Get Answer';
    }
}
